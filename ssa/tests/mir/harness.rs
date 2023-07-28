use colorz::{ansi::AnsiColor, Colorize};
use rayon::prelude::*;
use std::{
    collections::{BinaryHeap, LinkedList},
    ffi::OsStr,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

#[linkme::distributed_slice]
pub static TESTS: [TestSource] = [..];

#[macro_export]
macro_rules! test_source {
    ($($items:tt)*) => {
        #[allow(clippy::needless_update)]
        const _: () = {
            #[allow(unused)]
            use $crate::mir::harness::{MaybeIgnore, matches_extension};

            #[linkme::distributed_slice($crate::mir::harness::TESTS)]
            static TEST_SOURCE: $crate::mir::harness::TestSource = $crate::mir::harness::TestSource {
                $($items)*,
                ..$crate::mir::harness::TEST_SOURCE_DEFAULT
            };
        };
    };
}

pub fn matches_extension(path: &Path, ext: &str) -> bool {
    path.extension() == Some(OsStr::new(ext))
}

pub const TEST_SOURCE_DEFAULT: TestSource = TestSource {
    name: "",
    test_corpus: "",
    expected_path: |_| panic!("`expected_path` wasn't implemented"),
    path_matcher: |_| false,
    new_expected_path: None,
    test_runner: |_| unreachable!("no paths match, so no tests can run"),
};

#[doc(hidden)]
#[derive(Debug)]
pub struct RequriedTestSourceFields {
    pub test_corpus: (),
    pub expected_path: (),
    pub path_matcher: (),
    pub test_runner: (),
    pub new_expected_path: (),
}

#[doc(hidden)]
#[derive(Debug)]
pub struct TestSource {
    pub name: &'static str,
    pub test_corpus: &'static str,
    pub expected_path: fn(&Path) -> PathBuf,
    pub new_expected_path: Option<fn(&Path) -> PathBuf>,
    pub path_matcher: fn(&Path) -> bool,
    pub test_runner: fn(&TestInfo) -> std::io::Result<MaybeIgnore>,
}

#[allow(unused)]
#[derive(Debug, PartialEq, Eq)]
pub enum MaybeIgnore {
    Ran(String),
    Failed(String),
    Ignore,
    Skip,
}

#[derive(Default, Clone, Copy)]
struct Count {
    value: u32,
}

impl Count {
    fn print(&self, name: colorz::StyledValue<&'static str, impl colorz::OptionalColor>) {
        if self.value != 0 {
            println!("{name} {}", self.value)
        }
    }
}

macro_rules! Counts {
    (
        $($name:ident)*
    ) => {
        #[derive(Default, Clone, Copy)]
        struct Counts {
            $($name: Count,)*
        }

        impl core::ops::Add for Counts {
            type Output = Self;

            fn add(mut self, other: Self) -> Self {
                self += other;
                self
            }
        }

        impl core::ops::AddAssign for Counts {
            fn add_assign(&mut self, other: Self) {
                $(self.$name.value += other.$name.value;)*
            }
        }

        impl Counts {
            $(fn $name() -> Self {
                Self {
                    $name: Count { value: 1 },
                    ..Self::default()
                }
            })*
        }
    };
}

Counts! {
    regen
    save
    fail
    new
    pass
    ignore
    skip
}

#[derive(Debug)]
enum TestResult {
    Regen,
    Save,
    Fail { found: String },
    New(String),
    Pass,
    Ignore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TestResultKind {
    Regen,
    Save,
    Fail,
    New,
    Pass,
    Ignore,
}

#[derive(Debug)]
struct TestOutput {
    res: TestResult,
    test: TestInfo,
}

impl TestResult {
    fn kind(&self) -> TestResultKind {
        match self {
            TestResult::Ignore => TestResultKind::Ignore,
            TestResult::Pass => TestResultKind::Pass,
            TestResult::New(_) => TestResultKind::New,
            TestResult::Fail { .. } => TestResultKind::Fail,
            TestResult::Regen => TestResultKind::Regen,
            TestResult::Save => TestResultKind::Save,
        }
    }
}

impl Eq for TestOutput {}
impl PartialEq for TestOutput {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl PartialOrd for TestOutput {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TestOutput {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.res
            .kind()
            .cmp(&other.res.kind())
            .then_with(|| {
                self.test
                    .expected
                    .is_some()
                    .cmp(&other.test.expected.is_some())
            })
            .then_with(|| self.test.path.cmp(&other.test.path))
    }
}

#[derive(Debug)]
struct TestGroup<'a> {
    output: &'a TestOutput,
    rest: std::slice::Iter<'a, TestOutput>,
}

impl Eq for TestGroup<'_> {}
impl PartialEq for TestGroup<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl PartialOrd for TestGroup<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TestGroup<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.output.cmp(self.output)
    }
}

#[derive(Debug)]
pub struct TestInfo {
    pub path: PathBuf,
    expected_path: PathBuf,
    expected: Option<String>,
    source: &'static TestSource,
}

fn read_var(var: &str) -> bool {
    match std::env::var_os(var) {
        Some(x) => x == "1",
        None => false,
    }
}

pub fn run_tests() -> std::io::Result<()> {
    let start = std::time::Instant::now();

    println!("    Collecting tests...");

    let test_save = read_var("TEST_SAVE");
    let test_regen = read_var("TEST_REGEN");

    colorz::mode::set_coloring_mode_from_env();

    let mut tests = Vec::new();

    let mut parents = hashbrown::HashSet::new();

    for source in TESTS {
        for entry in walkdir::WalkDir::new(source.test_corpus) {
            let entry = entry?;

            if !(source.path_matcher)(entry.path()) {
                continue;
            }

            let expected_path = (source.expected_path)(entry.path());

            if test_save {
                if let Some(parent) = expected_path.parent() {
                    if !parents.contains(parent) {
                        parents.insert(parent.to_owned());
                        std::fs::create_dir_all(parent)?;
                    }
                }
            }

            tests.push(TestInfo {
                path: entry.path().to_owned(),
                expected_path,
                expected: None,
                source,
            })
        }
    }

    println!("    Collected {} tests...", tests.len());

    const DOTS_PER_LINE: u32 = 50;
    let lock = Mutex::new(0);
    let print_dot = |color: colorz::StyledValue<&str, AnsiColor>| {
        let mut lock = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        if *lock == DOTS_PER_LINE {
            *lock = 0;
            println!()
        }

        if *lock == 0 {
            print!("        ");
        }

        print!("{color}");

        *lock += 1;
    };

    let (test_groups, counts) = tests
        .into_par_iter()
        .map(|mut test| {
            test.expected = std::fs::read_to_string(&test.expected_path)
                .map(Some)
                .or_else(|err| {
                    if err.kind() == std::io::ErrorKind::NotFound {
                        Ok(None)
                    } else {
                        Err(err)
                    }
                })?;

            match (test.source.test_runner)(&test) {
                Err(err) => Err(err),
                Ok(MaybeIgnore::Ignore) => {
                    print_dot(".".fg(AnsiColor::BrightBlack));

                    Ok((
                        Some(TestOutput {
                            res: TestResult::Ignore,
                            test,
                        }),
                        Counts::ignore(),
                    ))
                }
                Ok(MaybeIgnore::Skip) => Ok((None, Counts::skip())),
                Ok(MaybeIgnore::Ran(output)) => {
                    let (res, counts) = match test.expected.as_deref() {
                        Some(expected) => {
                            if expected == output {
                                print_dot(".".fg(AnsiColor::BrightGreen));
                                (TestResult::Pass, Counts::pass())
                            } else if test_regen {
                                std::fs::write(&test.expected_path, output)?;
                                print_dot("R".fg(AnsiColor::Magenta));
                                (TestResult::Regen, Counts::regen())
                            } else {
                                print_dot("F".fg(AnsiColor::Red));
                                (TestResult::Fail { found: output }, Counts::fail())
                            }
                        }
                        None if test_save => {
                            std::fs::write(&test.expected_path, output)?;
                            print_dot("S".fg(AnsiColor::Blue));
                            (TestResult::Save, Counts::save())
                        }
                        None => {
                            print_dot("N".fg(AnsiColor::Cyan));
                            (TestResult::New(output), Counts::new())
                        }
                    };
                    Ok((Some(TestOutput { res, test }), counts))
                }
                Ok(MaybeIgnore::Failed(found)) => {
                    print_dot("F".fg(AnsiColor::Red));
                    Ok((
                        Some(TestOutput {
                            res: TestResult::Fail { found },
                            test,
                        }),
                        Counts::fail(),
                    ))
                }
            }
        })
        .try_fold(
            || (Vec::new(), Counts::default()),
            |(mut vec, counts), output| {
                output.map(|(output, c)| {
                    if let Some(output) = output {
                        vec.push(output);
                    }
                    (vec, counts + c)
                })
            },
        )
        .map(|output| {
            output.map(|(mut vec, counts)| {
                vec.sort_unstable();
                let mut list = LinkedList::new();
                list.push_back(vec);
                (list, counts)
            })
        })
        .try_reduce(
            Default::default,
            |(mut a_list, a_counts), (mut b_list, b_counts)| {
                a_list.append(&mut b_list);
                Ok((a_list, a_counts + b_counts))
            },
        )?;

    println!();
    println!();

    write_report(test_groups);

    let Counts {
        regen,
        save,
        fail,
        new,
        pass,
        ignore,
        skip,
    } = counts;

    skip.print("tests skipped".bright_black());
    ignore.print("tests ignored".bright_black());
    regen.print("tests regenerated".magenta());
    fail.print("tests failed".red());
    save.print("tests saved".blue());
    pass.print("tests passed".green());
    new.print("new tests".bright_cyan());

    if new.value != 0 || fail.value != 0 {
        println!()
    }

    if new.value != 0 {
        println!("To save all new tests, rerun tests with the TEST_SAVE env var set")
    }

    if fail.value != 0 {
        println!("To regnerate all failing tests, rerun tests with the TEST_REGEN env var set")
    }

    println!();
    println!("Running all tests took {:?}", start.elapsed());
    println!();

    if fail.value != 0 || new.value != 0 {
        std::process::exit(1);
    } else {
        std::process::exit(0);
    }
}

fn print_diff<W: io::Write>(writer: &mut W, error: &str, expected: &str) -> io::Result<()> {
    const EQUAL: colorz::Style<colorz::ansi::BrightBlack, colorz::NoColor, colorz::NoColor> =
        colorz::Style::new().fg(colorz::ansi::BrightBlack);

    const INSERT: colorz::Style<colorz::ansi::Green, colorz::NoColor, colorz::NoColor> =
        colorz::Style::new().fg(colorz::ansi::Green);

    const DELETE: colorz::Style<colorz::ansi::Red, colorz::NoColor, colorz::NoColor> =
        colorz::Style::new().fg(colorz::ansi::Red).strikethrough();

    let diff = dissimilar::diff(expected, error);

    for chunk in diff {
        match chunk {
            dissimilar::Chunk::Equal(x) => write!(writer, "{}", x.style_with(EQUAL))?,
            dissimilar::Chunk::Delete(x) => write!(writer, "{}", x.style_with(DELETE))?,
            dissimilar::Chunk::Insert(x) => write!(writer, "{}", x.style_with(INSERT))?,
        }
    }

    Ok(())
}

fn write_report(test_groups: LinkedList<Vec<TestOutput>>) {
    enum Section {
        None,
        Regen,
        Save,
    }

    let mut section = Section::None;
    let mut stdout = io::BufWriter::new(io::stdout().lock());

    macro_rules! println {
        ($($args:tt)*) => {
            writeln!(stdout, $($args)*).unwrap()
        };
    }

    // use a binary heap to merge the sorted lists easily
    let mut queue = BinaryHeap::with_capacity(test_groups.len());
    for test_group in test_groups.iter() {
        let mut tests = test_group.iter();
        if let Some(test) = tests.next() {
            queue.push(TestGroup {
                output: test,
                rest: tests,
            })
        }
    }

    while let Some(mut group) = queue.pop() {
        if let Some(test) = group.rest.next() {
            queue.push(TestGroup {
                output: test,
                rest: group.rest,
            });
        }

        let output = group.output;

        match &output.res {
            TestResult::Ignore | TestResult::Pass => break,
            TestResult::Regen => {
                println!(
                    "{} test: [{}] {}",
                    "Regenerate".magenta(),
                    output.test.source.name.bright_magenta(),
                    output.test.path.display().bright_magenta()
                );
                section = Section::Regen;
            }
            TestResult::Save => {
                if let Section::Save | Section::None = section {
                } else {
                    println!()
                }
                println!(
                    "{} test: [{}] {}",
                    "Save".blue(),
                    output.test.source.name.blue(),
                    output.test.path.display().blue()
                );
                section = Section::Save;
            }
            TestResult::New(found) => {
                if let Section::None = section {
                } else {
                    section = Section::None;
                    println!()
                }
                println!(
                    "{} test: [{}] {}",
                    "New".cyan(),
                    output.test.source.name.bright_cyan(),
                    output.test.path.display().bright_cyan()
                );

                let found = found.trim();
                println!("{}", found);
                println!()
            }
            TestResult::Fail { found } => {
                println!(
                    "Test {}: [{}] {}",
                    "failed".red(),
                    output.test.source.name.red(),
                    output.test.path.display().red()
                );

                let found = found.trim();
                if let Some(expected) = output.test.expected.as_deref() {
                    print_diff(&mut stdout, found, expected).unwrap()
                } else {
                    println!("{}", found);
                }
                println!()
            }
        }
    }

    if let Section::None = section {
    } else {
        println!()
    }
}
