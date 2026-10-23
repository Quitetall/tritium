//! Disposable bounded anonymous checkpoint storage, never recursive cleanup.
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
    cell::RefCell,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    rc::Rc,
};

struct State {
    root: PathBuf,
    identity: Option<Handle>,
    bytes: u64,
    max: u64,
    live: u64,
    files: u64,
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
pub(super) struct Workspace(Rc<RefCell<State>>);
pub(super) struct Report {
    pub cleaned: bool,
    pub peak_reserved_bytes: u64,
    pub peak_files: u64,
    pub warnings: Vec<String>,
}
impl Workspace {
    pub(super) fn new(root: PathBuf, bytes: u64, max: u64, updates: u64) -> io::Result<Self> {
        let required = bytes
            .checked_mul(if updates == 0 { 1 } else { 2 })
            .ok_or_else(|| io::Error::other("paged verification workspace byte overflow"))?;
        if bytes == 0 || required > max {
            return Err(io::Error::other(
                "paged verification workspace byte cap exceeded",
            ));
        }
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
        Ok(Self(Rc::new(RefCell::new(State {
            root,
            identity: Some(identity),
            bytes,
            max,
            live: 0,
            files: 0,
            peak: 0,
            peak_files: 0,
            next: 0,
            finished: false,
            cleaned: false,
            warnings: Vec::new(),
        }))))
    }
    pub(super) fn create(&mut self, ordinal: u64) -> io::Result<Checkpoint> {
        let mut s = self.0.borrow_mut();
        s.check()?;
        if ordinal != s.next {
            return Err(io::Error::other(
                "paged verification workspace ordinal differs",
            ));
        }
        let live = s
            .live
            .checked_add(s.bytes)
            .filter(|n| *n <= s.max)
            .ok_or_else(|| io::Error::other("paged verification workspace byte cap exceeded"))?;
        if s.files >= 2 {
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
            bytes: s.bytes,
            state: self.0.clone(),
        })
    }
    pub(super) fn finish(&mut self) -> Report {
        let mut s = self.0.borrow_mut();
        if !s.finished {
            if s.files != 0 {
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
        Report {
            cleaned: s.cleaned,
            peak_reserved_bytes: s.peak,
            peak_files: s.peak_files,
            warnings: s.warnings.clone(),
        }
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}
pub(super) struct Checkpoint {
    file: Option<File>,
    bytes: u64,
    state: Rc<RefCell<State>>,
}
impl Read for Checkpoint {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        if b.len() > 8192 {
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
        if b.len() > 8192 {
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
        let mut s = self.state.borrow_mut();
        s.live -= self.bytes;
        s.files -= 1;
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
        let mut w = Workspace::new(p.clone(), 256, 512, 1).unwrap();
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
}
