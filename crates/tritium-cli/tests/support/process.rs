//! Portable deadlines for direct CLI test children, with concurrent pipe drains.
use std::{
    io::{self, Read},
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

const MAX_CAPTURE: usize = 4 * 1024 * 1024;

fn capture(reader: impl Read) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take((MAX_CAPTURE + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_CAPTURE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "child output exceeds test capture limit",
        ));
    }
    Ok(bytes)
}

pub(crate) fn output(command: &mut Command, limit: Duration) -> io::Result<Output> {
    let deadline = Instant::now()
        .checked_add(limit)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid child deadline"))?;
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    thread::scope(|scope| {
        let stdout = scope.spawn(move || capture(stdout));
        let stderr = scope.spawn(move || capture(stderr));
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(5)),
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "CLI test child exceeded deadline",
                    ));
                }
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(error);
                }
            }
        };
        let stdout = stdout
            .join()
            .map_err(|_| io::Error::other("stdout reader panicked"))?;
        let stderr = stderr
            .join()
            .map_err(|_| io::Error::other("stderr reader panicked"))?;
        Ok(Output {
            status: status?,
            stdout: stdout?,
            stderr: stderr?,
        })
    })
}
