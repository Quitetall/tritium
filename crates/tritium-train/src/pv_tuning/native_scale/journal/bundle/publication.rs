//! New-directory single-file hard-link commit; no recursive cleanup or overwrite.
use super::*;
use std::{
    fs::{self, File, OpenOptions},
    path::Path,
};
use tritium_quantize::engine::PublicationWarning;

/// Committed recovery inputs with explicit post-commit cleanup/sync caveats.
#[derive(Debug)]
pub struct PublishedNativeScaleResumeBundle {
    /// Retained byte-consistent inputs, not numerical or quality qualification.
    pub bundle: NativeScaleResumeBundleReader<File>,
    /// No post-commit cleanup or directory-sync failure is reported as failed commit.
    pub warnings: Vec<PublicationWarning>,
}

/// Publish `resume.tnsb` in a new directory without replacing any existing path.
/// Fsync precedes the hard-link commit. Caller owns parent directory/source
/// custody and aggregate disk/memory admission. Not power-loss/security qualification.
pub fn publish_native_scale_resume_bundle<W: Read + Write + Seek>(
    session: &NativeScaleRefinementSession,
    journal: &mut NativeScaleJournalWriter<W>,
    directory: &Path,
    limits: NativeScaleResumeBundleLimits,
) -> Result<PublishedNativeScaleResumeBundle> {
    let prepared = stage::prepare(journal, session, limits)?;
    fs::create_dir(directory)?;
    let pending = directory.join(".pending.tnsb");
    let destination = directory.join("resume.tnsb");
    let mut owned = false;
    let result = (|| {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&pending)?;
        owned = true;
        let mut bundle = stage::stage(prepared, journal, file, limits)?;
        bundle.source.sync_all()?;
        bundle.verify_unchanged()?;
        fs::hard_link(&pending, &destination)?;
        Ok(bundle)
    })();
    let bundle = match result {
        Ok(bundle) => bundle,
        Err(error) => {
            if owned {
                let _ = fs::remove_file(&pending);
            }
            // Unexpected/foreign files must survive a failed publication.
            let _ = fs::remove_dir(directory);
            return Err(error);
        }
    };
    let mut warnings = Vec::new();
    if fs::remove_file(&pending).is_err() {
        warnings.push(PublicationWarning::PendingCleanup);
    }
    let parent = directory
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if File::open(directory).and_then(|f| f.sync_all()).is_err()
        || File::open(parent).and_then(|f| f.sync_all()).is_err()
    {
        warnings.push(PublicationWarning::DirectorySync);
    }
    Ok(PublishedNativeScaleResumeBundle { bundle, warnings })
}
