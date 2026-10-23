//! Shared bounded anonymous work storage, never recursive cleanup.
use same_file::Handle;
#[derive(Debug)]
struct Initialization {
    source: io::Error,
    root: PathBuf,
}
impl std::fmt::Display for Initialization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}; new workspace retained at {} because identity is unavailable",
            self.source,
            self.root.display()
        )
    }
}
impl std::error::Error for Initialization {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
use std::{
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
};

struct State {
    root: PathBuf,
    identity: Option<Handle>,
    bytes: u64,
    max: u64,
    live: u64,
    files: u64,
    max_files: u64,
    peak: u64,
    peak_files: u64,
    next: u64,
    finished: bool,
    cleaned: bool,
    warnings: Vec<String>,
}
impl State {
    fn check(&self) -> io::Result<()> {
        if self.finished {
            return Err(io::Error::other("paged verification workspace is closed"));
        }
        let meta = fs::symlink_metadata(&self.root)?;
        if !meta.is_dir()
            || meta.file_type().is_symlink()
            || self.identity.as_ref() != Some(&Handle::from_path(&self.root)?)
        {
            return Err(io::Error::other(
                "paged verification workspace identity changed",
            ));
        }
        Ok(())
    }
}
pub(crate) struct Workspace(Arc<Mutex<State>>);
pub(crate) struct Report {
    pub reserved_bytes: u64,
    pub files: u64,
    pub cleaned: bool,
    pub peak_reserved_bytes: u64,
    pub peak_files: u64,
    pub warnings: Vec<String>,
}
impl Workspace {
    pub(crate) fn report(&self) -> Report {
        let s = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut r = report(&s);
        if self.0.is_poisoned() {
            r.warnings
                .push("WorkspaceCleanup: lock poisoned; custody/accounting unqualified".into());
        }
        r
    }
    pub(crate) fn new(root: PathBuf, bytes: u64, max: u64, updates: u64) -> io::Result<Self> {
        let required = bytes
            .checked_mul(if updates == 0 { 1 } else { 2 })
            .ok_or_else(|| io::Error::other("paged verification workspace byte overflow"))?;
        if bytes == 0 || required > max {
            return Err(io::Error::other(
                "paged verification workspace byte cap exceeded",
            ));
        }
        Self::initialize(root, bytes, max, 2)
    }
    /// Private generalized storage seam; callers preflight role-specific extents.
    pub(crate) fn storage(root: PathBuf, max: u64, max_files: u64) -> io::Result<Self> {
        if max == 0 || max_files == 0 {
            return Err(io::Error::other("paged workspace limits must be positive"));
        }
        Self::initialize(root, 0, max, max_files)
    }
    fn initialize(root: PathBuf, bytes: u64, max: u64, max_files: u64) -> io::Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            fs::DirBuilder::new().mode(0o700).create(&root)?;
        }
        #[cfg(not(unix))]
        fs::create_dir(&root)?;
        let identity = match Handle::from_path(&root) {
            Ok(v) => v,
            Err(e) => {
                // No identity means no authority to remove this path.
                return Err(io::Error::new(e.kind(), Initialization { source: e, root }));
            }
        };
        Ok(Self(Arc::new(Mutex::new(State {
            root,
            identity: Some(identity),
            bytes,
            max,
            live: 0,
            files: 0,
            max_files,
            peak: 0,
            peak_files: 0,
            next: 0,
            finished: false,
            cleaned: false,
            warnings: Vec::new(),
        }))))
    }
    pub(crate) fn create(&mut self, ordinal: u64) -> io::Result<Checkpoint> {
        let bytes = self.0.lock().map_err(poison)?.bytes;
        self.allocate(Some(ordinal), bytes, 8192)
    }
    pub(crate) fn create_blob(&self, bytes: u64, io_bytes: usize) -> io::Result<Checkpoint> {
        self.allocate(None, bytes, io_bytes)
    }
    fn allocate(
        &self,
        ordinal: Option<u64>,
        bytes: u64,
        io_bytes: usize,
    ) -> io::Result<Checkpoint> {
        let mut s = self.0.lock().map_err(poison)?;
        s.check()?;
        if ordinal.is_some_and(|i| i != s.next) {
            return Err(io::Error::other(
                "paged verification workspace ordinal differs",
            ));
        }
        let live = s
            .live
            .checked_add(bytes)
            .filter(|n| *n <= s.max)
            .ok_or_else(|| io::Error::other("paged verification workspace byte cap exceeded"))?;
        if bytes == 0 || io_bytes == 0 || s.files >= s.max_files {
            return Err(io::Error::other(
                "paged verification workspace file cap exceeded",
            ));
        }
        let next = s
            .next
            .checked_add(1)
            .ok_or_else(|| io::Error::other("paged verification workspace ordinal overflow"))?;
        let file = tempfile::tempfile_in(&s.root)?;
        // The private directory must still be the admitted directory after creation.
        s.check()?;
        s.live = live;
        s.files += 1;
        s.peak = s.peak.max(live);
        s.peak_files = s.peak_files.max(s.files);
        s.next = next;
        Ok(Checkpoint {
            file: Some(file),
            bytes,
            io_bytes,
            state: self.0.clone(),
        })
    }
    pub(crate) fn finish(&self) -> Report {
        let mut s = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !s.finished {
            if self.0.is_poisoned() {
                s.warnings
                    .push("WorkspaceCleanup: lock poisoned; directory preserved".into());
            } else if s.files != 0 {
                s.warnings
                    .push("workspace has live checkpoint leases".into());
            } else if let Err(e) = s.check() {
                s.warnings.push(format!("WorkspaceCleanup: {e}"));
            } else {
                // Close the identity handle for platforms requiring it before rmdir.
                // Caller-exclusive custody is required; this is not a security seal.
                drop(s.identity.take());
                match fs::remove_dir(&s.root) {
                    Ok(()) => s.cleaned = true,
                    Err(e) => s.warnings.push(format!("WorkspaceCleanup: {e}")),
                }
            }
            s.finished = true;
        }
        report(&s)
    }
}
fn report(s: &State) -> Report {
    Report {
        reserved_bytes: s.live,
        files: s.files,
        cleaned: s.cleaned,
        peak_reserved_bytes: s.peak,
        peak_files: s.peak_files,
        warnings: s.warnings.clone(),
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}
fn poison<T>(_: std::sync::PoisonError<T>) -> io::Error {
    io::Error::other("paged workspace lock poisoned")
}
pub(crate) struct Checkpoint {
    file: Option<File>,
    bytes: u64,
    io_bytes: usize,
    state: Arc<Mutex<State>>,
}
impl Read for Checkpoint {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        if b.len() > self.io_bytes {
            return Err(io::Error::other("paged checkpoint I/O window exceeded"));
        }
        self.file.as_mut().expect("live checkpoint").read(b)
    }
}
impl Seek for Checkpoint {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        self.file.as_mut().expect("live checkpoint").seek(s)
    }
}
impl Write for Checkpoint {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if b.len() > self.io_bytes {
            return Err(io::Error::other("paged checkpoint I/O window exceeded"));
        }
        let file = self.file.as_mut().expect("live checkpoint");
        if file
            .stream_position()?
            .checked_add(b.len() as u64)
            .is_none_or(|n| n > self.bytes)
        {
            return Err(io::Error::other("paged checkpoint extent cap exceeded"));
        }
        file.write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.as_mut().expect("live checkpoint").flush()
    }
}
impl Drop for Checkpoint {
    fn drop(&mut self) {
        drop(self.file.take());
        let mut s = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        s.live = s.live.saturating_sub(self.bytes);
        s.files = s.files.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn root() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }
    #[test]
    fn quota_extent_ordinal_and_anonymous_retirement_are_exact() {
        let root = root();
        let p = root.path().join("work");
        assert!(Workspace::new(p.clone(), 256, 511, 1).is_err());
        assert!(Workspace::new(p.clone(), u64::MAX, u64::MAX, 1).is_err());
        assert!(!p.exists());
        let mut w = Workspace::new(p.clone(), 256, 512, 1).unwrap();
        let mut a = w.create(0).unwrap();
        let b = w.create(1).unwrap();
        assert!(w.create(2).is_err());
        assert!(a.write_all(&[0; 257]).is_err());
        a.write_all(&[1; 256]).unwrap();
        assert!(a.write_all(&[0]).is_err());
        assert!(a.read(&mut [0; 8193]).is_err());
        drop(a);
        let c = w.create(2).unwrap();
        drop(b);
        drop(c);
        let r = w.finish();
        assert!(r.cleaned);
        assert!(r.warnings.is_empty());
        assert_eq!(r.peak_reserved_bytes, 512);
        assert_eq!(r.peak_files, 2);
        assert!(!p.exists());
    }
    #[test]
    fn existing_and_unexpected_files_are_never_removed() {
        let root = root();
        let p = root.path().join("work");
        fs::create_dir(&p).unwrap();
        fs::write(p.join("foreign"), b"keep").unwrap();
        assert!(Workspace::new(p.clone(), 256, 512, 1).is_err());
        assert_eq!(fs::read(p.join("foreign")).unwrap(), b"keep");
        let p = root.path().join("owned");
        let w = Workspace::new(p.clone(), 256, 512, 1).unwrap();
        fs::write(p.join("foreign"), b"keep").unwrap();
        let r = w.finish();
        assert!(!r.cleaned);
        assert!(!r.warnings.is_empty());
        assert_eq!(fs::read(p.join("foreign")).unwrap(), b"keep");
    }
    #[test]
    fn unwind_releases_anonymous_files_and_only_own_empty_directory() {
        let root = root();
        let p = root.path().join("work");
        let result = std::panic::catch_unwind(|| {
            let mut w = Workspace::new(p.clone(), 256, 512, 1).unwrap();
            let _a = w.create(0).unwrap();
            panic!("injected workspace unwind");
        });
        assert!(result.is_err());
        assert!(!p.exists());
    }
    #[test]
    fn replaced_directory_is_not_reused_or_removed() {
        let root = root();
        let p = root.path().join("work");
        let moved = root.path().join("moved");
        let mut w = Workspace::new(p.clone(), 256, 512, 1).unwrap();
        fs::rename(&p, &moved).unwrap();
        fs::create_dir(&p).unwrap();
        assert!(w.create(0).is_err());
        assert!(!w.finish().cleaned);
        assert!(p.exists());
        assert!(moved.exists());
    }
    #[test]
    fn role_extents_io_quota_and_cross_thread_retirement_share_one_account() {
        let root = root();
        let path = root.path().join("roles");
        let w = Workspace::storage(path.clone(), 300, 2).unwrap();
        let a = w.create_blob(100, 8).unwrap();
        let mut b = w.create_blob(200, 16).unwrap();
        assert!(w.create_blob(1, 8).is_err());
        assert_eq!((w.report().reserved_bytes, w.report().files), (300, 2));
        assert!(b.write_all(&[0; 17]).is_err());
        std::thread::spawn(move || {
            let mut a = a;
            a.write_all(&[1; 8]).unwrap();
            a.seek(SeekFrom::Start(99)).unwrap();
            assert!(a.write_all(&[0; 2]).is_err());
        })
        .join()
        .unwrap();
        assert_eq!((w.report().reserved_bytes, w.report().files), (200, 1));
        let c = w.create_blob(100, 1).unwrap();
        assert!(w.create_blob(1, 1).is_err());
        drop(b);
        drop(c);
        let r = w.finish();
        assert!(r.cleaned);
        assert_eq!(r.peak_reserved_bytes, 300);
        assert_eq!(r.peak_files, 2);
        assert!(!path.exists());
    }
    #[test]
    fn poisoned_storage_refuses_admission_and_preserves_owned_directory() {
        let root = root();
        let path = root.path().join("poison");
        let w = Workspace::storage(path.clone(), 100, 1).unwrap();
        let a = w.create_blob(100, 8).unwrap();
        assert!(
            std::panic::catch_unwind(|| {
                let _guard = w.0.lock().unwrap();
                panic!("injected storage poison");
            })
            .is_err()
        );
        assert!(w.create_blob(1, 1).is_err());
        drop(a);
        let r = w.finish();
        assert!(!r.cleaned);
        assert_eq!(r.files, 0);
        assert!(r.warnings.iter().any(|s| s.contains("poisoned")));
        assert!(path.exists());
    }
}
