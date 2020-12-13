import argparse
import copy
import csv
import datetime
import itertools
import statistics
import sys
from typing import Any, Callable, Dict, List, Optional, Set, TextIO, Tuple

from student import Assignment, Category, GradeReport, Multiplier, Student

# TODO Put this in a config.
LATE_MULTIPLIER_DESC = 'Late multiplier'
LATE_MULTIPLIERS = [0.9, 0.8, 0.6]
LATE_GRACE = datetime.timedelta(minutes=5)
VERBOSE_COMMENTS = True

def import_categories(path: str) -> Dict[str, Category]:
    """Imports assignment categories the CSV file at the given path and initializes students' slip day and drop values.

    :param path: The path of the category CSV.
    :type path: str
    :returns: A dict mapping category names to categories.
    :rtype: dict
    """
    categories: Dict[str, Category] = {}
    with open(path) as roster_file:
        reader = csv.DictReader(roster_file)
        for row in reader:
            name = row['Name']
            weight = float(row['Weight'])
            has_late_multiplier = bool(int(row['Has Late Multiplier']))
            drops = int(row['Drops'])
            slip_days = int(row['Slip Days'])
            categories[name] = Category(name, weight, drops, slip_days, has_late_multiplier)
    return categories

def import_assignments(path: str, categories: Dict[str, Category]) -> Dict[str, Assignment]:
    """Imports assignments from the CSV file at the given path.

    :param path: The path of the assignments CSV.
    :type path: str
    :param categories: The categories for assignments.
    :type categories: dict
    :returns: A dict mapping assignment names to assignments.
    :rtype: dict
    """
    assignments: Dict[str, Assignment] = {}
    with open(path) as assignment_file:
        reader = csv.DictReader(assignment_file)
        for row in reader:
            name = row['Name']
            category = row['Category']
            score_possible = float(row['Possible'])
            weight = float(row['Weight'])
            slip_group = int(row['Slip Group'])
            if category not in categories:
                raise RuntimeError(f'Assignment {name} references unknown category {category}')
            assignments[name] = Assignment(name, category, score_possible, weight, slip_group)
    return assignments

def import_overrides(path: str, assignments: Dict[str, Assignment]) -> Dict[int, Dict[str, float]]:
    """Imports overrides from the CSV file at the given path.

    :param path: The path of the overrides CSV.
    :type path: str
    :param assignments: The assignments being overridden.
    :type assignments: dict
    :returns: A nested dict mapping SIDs to assignment names to overridden scores.
    :rtype: dict
    """
    overrides: Dict[int, Dict[str, float]] = {}
    with open(path) as override_file:
        reader = csv.DictReader(override_file)
        for row in reader:
            sid = int(row['SID'])
            assignment_name = row['Assignment']
            score = float(row['Score'])
            if assignment_name not in assignments:
                raise RuntimeError(f'Override for {sid} references unknown assignment {assignment_name}')
            overrides.setdefault(sid, {})
            if assignment_name in overrides[sid]:
                raise RuntimeError(f'Duplicate override of {assignment_name} for {sid}')
            overrides[sid][assignment_name] = score
    return overrides

def import_roster_and_grades(roster_path: str, grades_path: str, categories: Dict[str, Category], assignments: Dict[str, Assignment], overrides: Optional[Dict[int, Dict[str, float]]] = None) -> Dict[int, List[Student]]:
    """Imports the CalCentral roster in the CSV file at the given path and then initializes students with the grades present in the given Gradescope grade report.

    :param roster_path: The path of the CalCentral roster.
    :type roster_path: str
    :param grades_path: The path of the Gradescope grade report.
    :type grades_path: str
    :param categories: The categories to initialize the students with.
    :type categories: dict
    :param assignments: The assignments to initialize the students with.
    :type assignments: dict
    :param overrides: Grade overrides for individual students.
    :type overrides: dict
    :returns: A dict mapping student IDs to a one-element list of students.
    :rtype: dict
    """
    students: Dict[int, List[Student]] = {}
    with open(roster_path) as roster_file:
        reader = csv.DictReader(roster_file)
        for row in reader:
            sid = int(row['Student ID'])
            name = row['Name']
            students[sid] = [Student(sid, name, categories, assignments)]
    with open(grades_path) as grades_file:
        reader = csv.DictReader(grades_file)
        not_present_names: Set[str] = set()
        for row in reader:
            try:
                sid = int(row['SID'])
            except ValueError as e:
                continue
            if sid not in students:
                # Skip students not in roster.
                continue

            # Create the base dict of student assignments.
            student_assignments = copy.deepcopy(assignments)
            for assignment in student_assignments.values():
                assignment_lateness_header = f'{assignment.name} - Lateness (H:M:S)'
                assignment_max_points_header = f'{assignment.name} - Max Points'

                score: float
                comments: List[str] = []
                if assignment.name in row:
                    scorestr = row[assignment.name]
                    if scorestr != '':
                        score = float(scorestr)
                        # Lateness formatted as HH:MM:SS.
                        lateness_components = row[assignment_lateness_header].split(':')
                        hours = int(lateness_components[0])
                        minutes = int(lateness_components[1])
                        seconds = int(lateness_components[2])
                        lateness = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)

                        # Take min with max score possible on Gradescope.
                        max_score = float(row[assignment_max_points_header])
                        score = min(max_score, score)

                        if overrides is not None and sid in overrides and assignment.name in overrides[sid]:
                            # Overridden.
                            new_score = overrides[sid][assignment.name]
                            comments.append(f'Overridden from {score}/{assignment.score_possible} to {new_score}/{assignment.score_possible}')
                            score = new_score
                    else:
                        # Empty string score string means no submission; assume 0.0.
                        score = 0.0
                        lateness = datetime.timedelta(0)
                else:
                    # No column for assignment; assume 0.0.
                    score = 0.0
                    lateness = datetime.timedelta(0)
                    if assignment.name not in not_present_names:
                        not_present_names.add(assignment.name)
                        print(f'Warning: No grades present for {assignment.name}', file=sys.stderr)
                student_assignments[assignment.name].grade = Assignment.Grade(score, lateness, comments=comments)

            # Copy this dict to each student.
            for student in students[sid]:
                student.assignments = copy.deepcopy(student_assignments)
    return students

def apply_policy(policy: Callable[[Student], List[Student]], students: Dict[int, List[Student]]) -> Dict[int, List[Student]]:
    """Applies a policy function by flat mapping the returned list of outputs for each input student into a new iterable and returning it.

    :param policy: The policy function to apply.
    :type policy: callable
    :param students: The input students.
    :type students: list
    :returns: The SIDs mapped to the new students.
    :rtype: dict
    """
    new_students: Dict[int, List[Student]] = {}
    for sid in students.keys():
        new_students[sid] = [new_student for student in students[sid] for new_student in policy(student)]
        assert len(new_students[sid]) > 0, 'Policy function returned an empty list'
    return new_students

def make_accommodations(path: str) -> Callable[[Student], List[Student]]:
    """Returns a policy function that applies the accommodations in the CSV at the given path.

    Accommodations are applied by mutating the student objects to adjust how many drops and slip days they have.

    :param path: The path of the accommodations CSV.
    :type path: str
    :returns: An accommodations policy function.
    :rtype: callable
    """
    accommodations: Dict[int, List[Dict[str, str]]] = {}
    with open(path) as accommodations_file:
        reader = csv.DictReader(accommodations_file)
        for row in reader:
            sid = int(row['SID'])
            accommodations.setdefault(sid, []).append(row)

    def accommodations_policy(student: Student) -> List[Student]:
        if student.sid not in accommodations:
            return [student]
        new_student = copy.deepcopy(student)
        for row in accommodations[new_student.sid]:
            category = row['Category']
            drop_adjust = int(row['Drop Adjust'])
            slip_day_adjust = int(row['Slip Day Adjust'])
            if category not in student.categories:
                # If not present, it wasn't present in categories CSV.
                raise RuntimeError(f'Accommodations reference nonexistent category {category}')
            new_student.categories[category].drops += drop_adjust
            new_student.categories[category].slip_days += slip_day_adjust
            if VERBOSE_COMMENTS:
                if drop_adjust > 0:
                    new_student.categories[category].comments.append(f'{drop_adjust} extra drops granted')
                if slip_day_adjust > 0:
                    new_student.categories[category].comments.append(f'{slip_day_adjust} extra drops granted')
        return [new_student]
    return accommodations_policy

def make_extensions(path: str) -> Callable[[Student], List[Student]]:
    """Returns a policy function that applies the extensions in the CSV file.

    :param path: The path of the extensions CSV.
    :type path: str
    :returns: An extensions policy function.
    :rtype: callable
    """
    extensions: Dict[int, List[Dict[str, str]]] = {}
    with open(path) as extensions_file:
        reader = csv.DictReader(extensions_file)
        for row in reader:
            sid = int(row['SID'])
            extensions.setdefault(sid, []).append(row)
    zero = datetime.timedelta(0)

    def extensions_policy(student: Student) -> List[Student]:
        if student.sid not in extensions:
            return [student]
        new_student = copy.deepcopy(student)
        for row in extensions[new_student.sid]:
            assignment_name = row['Assignment']
            days = int(row['Days'])
            if assignment_name not in student.assignments:
                # If not present, it wasn't present in assignments CSV.
                raise RuntimeError(f'Extension references unknown assignment {assignment_name}')
            grade = new_student.assignments[assignment_name].grade
            grade.lateness = max(grade.lateness - datetime.timedelta(days=days), zero)
            if VERBOSE_COMMENTS:
                grade.comments.append(f'{days}-day extension granted')
        return [new_student]
    return extensions_policy

def slip_day_policy(student: Student) -> List[Student]:
    """A policy function that applies slip days per category.

    Slip days are applied using a brute-force method of enumerating all possible ways to assign slip days to assignments. The appropriate lateness is removed from the grade entry, and a comment is added.

    :param student: The input student.
    :type student: Student
    :returns: A list of students containing all possibile ways of applying slip days.
    :rtype: list
    """
    def get_slip_possibilities(num_assignments: int, slip_days: int) -> List[List[int]]:
        # Basically np.meshgrid with max sum <= slip_days.
        # TODO Optimize by removing unnecessary slip day possiblities (e.g. only using 2 when you can use 3).
        if num_assignments == 0:
            return [[]]
        possibilities: List[List[int]] = []
        for i in range(slip_days + 1):
            # i is the number of slip days used for the first assignment.
            rest = get_slip_possibilities(num_assignments - 1, slip_days - i)
            rest = [[i] + possibility for possibility in rest]
            possibilities.extend(rest)
        return possibilities

    zero = datetime.timedelta(0)

    # slip_groups[i] have slip_possibilities[i].
    slip_groups: List[Set[int]] = []
    slip_possibilities: List[List[List[int]]] = []

    for category_name in student.categories:
        category = student.categories[category_name]
        # Get all slip groups that the student has late in the category.
        category_slip_groups: Set[int] = set()
        for assignment in student.assignments.values():
            if assignment.category == category.name and assignment.slip_group != -1 and assignment.grade.lateness > zero:
                category_slip_groups.add(assignment.slip_group)

        # Get all possible ways of applying slip days.
        category_slip_possibilities = get_slip_possibilities(len(category_slip_groups), category.slip_days)

        slip_groups.append(category_slip_groups)
        slip_possibilities.append(category_slip_possibilities)

    new_students: List[Student] = [student]

    # All possibilities is the cross product of all possibilities in each category.
    for slip_possibility in itertools.product(*slip_possibilities):
        if sum(slip_days for category_slip_possibility in slip_possibility for slip_days in category_slip_possibility) == 0:
            # Skip 0 slip day application case since it is already present in the list.
            continue
        student_with_slip = copy.deepcopy(student)
        for category_index in range(len(slip_possibility)):
            category_slip_groups = slip_groups[category_index]
            category_slip_groups_list = list(category_slip_groups)
            category_slip_possibility = slip_possibility[category_index]
            for i in range(len(category_slip_groups_list)):
                slip_group = category_slip_groups_list[i]
                slip_days = category_slip_possibility[i]
                if slip_days == 0:
                    # Not applying slip days for this group for this possibility, so skip.
                    continue
                student_with_slip.slip_days_used += slip_days
                for assignment in student_with_slip.assignments.values():
                    if assignment.slip_group == slip_group:
                        assignment.grade.lateness = max(assignment.grade.lateness - datetime.timedelta(days=slip_days), zero)
                        assignment.grade.slip_days_applied += slip_days
                        assignment.grade.comments.append(f"{slip_days} slip {'days' if slip_days > 1 else 'day'} applied")
        new_students.append(student_with_slip)

    return new_students

def late_multiplier_policy(student: Student) -> List[Student]:
    """A policy function that applies late multipliers.

    Late multipliers are applied by appending to each grade's multipliers list.

    :param student: The input student.
    :type student: Student
    :returns: A one-element list containing a student with late multipliers applied.
    :rtype: list
    """
    zero = datetime.timedelta(0)
    one = datetime.timedelta(days=1)

    def get_days_late(lateness: datetime.timedelta) -> int:
        lateness = max(zero, lateness)
        days_late = lateness.days
        if lateness % one > LATE_GRACE:
            days_late += 1
        return days_late

    new_student = copy.deepcopy(student)

    # Build dict mapping slip groups to maximal number of days late.
    slip_group_lateness: Dict[int, datetime.timedelta] = {}
    for assignment in new_student.assignments.values():
        if assignment.grade.lateness > zero and assignment.slip_group != -1 and (assignment.slip_group not in slip_group_lateness or assignment.grade.lateness > slip_group_lateness[assignment.slip_group]):
            slip_group_lateness[assignment.slip_group] = assignment.grade.lateness

    # Apply lateness.
    for assignment in new_student.assignments.values():
        category = student.categories[assignment.category]

        # Lateness is based on individual assignment if no slip group, else use early maximal value.
        days_late: int
        if assignment.slip_group in slip_group_lateness:
            days_late = get_days_late(slip_group_lateness[assignment.slip_group])
        else:
            days_late = get_days_late(assignment.grade.lateness)

        if days_late > 0:
            late_multipliers: List[float]
            if category.has_late_multiplier:
                late_multipliers = LATE_MULTIPLIERS
            else:
                # Empty array means immediately 0.0 upon late.
                late_multipliers = []

            if days_late <= len(late_multipliers): # <= because zero-indexing.
                multiplier = late_multipliers[days_late - 1] # + 1 because zero-indexing.
            else:
                # Student submitted past latest possible time.
                multiplier = 0.0
            assignment.grade.multipliers_applied.append(Multiplier(multiplier, LATE_MULTIPLIER_DESC))

    return [new_student]

def drop_policy(student: Student) -> List[Student]:
    """A policy function that applies drops per categories.

    Drops are applied by setting the dropped variable for all possible combinations of assignments to drop in each category.

    :param student: The input student.
    :type student: Student
    :returns: A list of students containing all possibilities of applying drops.
    :rtype: list
    """
    # Assignments in drop_assignments[i] have drop_possibilities[i].
    drop_assignments: List[List[str]] = []
    drop_possibilities: List[Tuple[Tuple[bool, ...], ...]] = []

    for category in student.categories.values():
        # Get all ways to assign drops to assignments in the category.
        drops = student.categories[category.name].drops
        assignments_in_category = [assignment for assignment in student.assignments.values() if assignment.category == category.name]
        category_possibility = tuple(i < drops for i in range(len(assignments_in_category)))

        drop_assignments.append([assignment.name for assignment in assignments_in_category])
        drop_possibilities.append(tuple(sorted(set(itertools.permutations(category_possibility)))))

    new_students: List[Student] = []
    for drop_possibility in itertools.product(*drop_possibilities):
        new_student = copy.deepcopy(student)
        for category_index in range(len(drop_possibility)):
            category_possibility = drop_possibility[category_index]
            for assignment_index in range(len(category_possibility)):
                assignment = new_student.assignments[drop_assignments[category_index][assignment_index]]
                if assignment.grade.slip_days_applied > 0:
                    # Skip possibility if we applied slip days, since we never want to do apply slip days to dropped assignments.
                    continue
                should_drop = category_possibility[assignment_index]
                if should_drop:
                    assignment.grade.dropped = True
                    assignment.grade.comments.append('Dropped')
        new_students.append(new_student)
    return new_students

def make_clobbers(path: str, category_names: List[str], assignment_names: List[str], students: Dict[int, List[Student]]) -> Callable[[Student], List[Student]]:
    """Returns a policy function that applies clobbers based on the statistics in preliminary grade reports.

    Clobbers are applied by returning every possibility of applying clobbers via assignment and category overrides.

    :param path: The path of the clobbers CSV.
    :type path: str
    :param students: The students from which to generate preliminary grade reports.
    :type students: dict
    :returns: A clobber policy function.
    :rtype: callable
    """
    def zscore_clobber(source_score: float, source_mean: float, source_stdev: float, target_mean: float, target_stdev: float) -> float:
        return target_mean + target_stdev * (source_score - source_stdev) / source_mean

    prelim_reports = generate_grade_reports(students)

    category_clobbers: Dict[int, Dict[str, Tuple[str, float]]] = {} # SID -> target name -> (source, score)
    assignment_clobbers: Dict[int, Dict[str, Tuple[str, float]]] = {}
    with open(path) as clobbers_file:
        reader = csv.DictReader(clobbers_file)
        for row in reader:
            scope = row['Scope']
            target = row['Target']
            source = row['Source']
            scale = float(row['Scale'])
            clobber_type = row['Type']
            for sid in prelim_reports:
                if scope == 'CATEGORY':
                    source_score = prelim_reports[sid].categories[source].adjusted
                elif scope == 'ASSIGNMENT':
                    source_score = prelim_reports[sid].assignments[source].adjusted
                else:
                    raise RuntimeError(f'Unknown clobber scope {scope}')
                if clobber_type == 'SCALED':
                    new_score = source_score
                elif clobber_type == 'ZSCORE':
                    if scope == 'CATEGORY':
                        source_mean = statistics.mean(report.categories[source].adjusted for report in prelim_reports.values())
                        source_stdev = statistics.stdev(report.categories[source].adjusted for report in prelim_reports.values())
                        target_mean = statistics.mean(report.categories[target].adjusted for report in prelim_reports.values())
                        target_stdev = statistics.stdev(report.categories[target].adjusted for report in prelim_reports.values())
                    else: # scope == 'ASSIGNMENT'
                        source_mean = statistics.mean(report.assignments[source].adjusted for report in prelim_reports.values())
                        source_stdev = statistics.stdev(report.assignments[source].adjusted for report in prelim_reports.values())
                        target_mean = statistics.mean(report.assignments[target].adjusted for report in prelim_reports.values())
                        target_stdev = statistics.stdev(report.assignments[target].adjusted for report in prelim_reports.values())
                    new_score = zscore_clobber(source_score, source_mean, source_stdev, target_mean, target_stdev)
                new_score *= scale
                if scope == 'CATEGORY':
                    category_clobbers.setdefault(sid, {}).setdefault(target, ('', -float('inf')))
                    if category_clobbers[sid][target][1] < new_score:
                        category_clobbers[sid][target] = (source, new_score)
                else:
                    assignment_clobbers.setdefault(sid, {}).setdefault(target, ('', -float('inf')))
                    if assignment_clobbers[sid][target][1] < new_score:
                        assignment_clobbers[sid][target] = (source, new_score)

    def get_binary_combinations(n: int) -> List[List[bool]]:
        if n == 0:
            return [[]]
        rest = get_binary_combinations(n - 1)
        ret = []
        for r in rest:
            ret.append([False, *r])
            ret.append([True, *r])
        return ret

    def clobber_policy(student: Student) -> List[Student]:
        # category_names[i] has combinations category_combinations[i].
        if student.sid in category_clobbers:
            category_names = [name for name in category_clobbers[student.sid] if student.categories[name].override is None]
        else:
            category_names = []
        category_combinations = get_binary_combinations(len(category_names))
        # assignment_names[i] has combinations assignment_combinations[i].
        if student.sid in assignment_clobbers:
            assignment_names = [name for name in assignment_clobbers[student.sid] if student.assignments[name].grade.override is None]
        else:
            assignment_names = []
        assignment_combinations = get_binary_combinations(len(assignment_names))

        new_students: List[Student] = []
        for combination in itertools.product(category_combinations, assignment_combinations):
            new_student = copy.deepcopy(student)
            category_combination = combination[0]
            assignment_combination = combination[1]
            for i in range(len(category_combination)):
                if category_combination[i]:
                    category_name = category_names[i]
                    category_clobber = category_clobbers[student.sid][category_name]
                    category = new_student.categories[category_name]
                    category.override = category_clobber[1]
                    category.comments.append(f'Clobbered by {category_clobber[0]}')
            for i in range(len(assignment_combination)):
                if assignment_combination[i]:
                    assignment_name = assignment_names[i]
                    assignment_clobber = assignment_clobbers[student.sid][assignment_name]
                    assignment = new_student.assignments[assignment_name]
                    assignment.grade.override = assignment_clobber[1] * assignment.score_possible
                    assignment.grade.comments.append(f'Clobbered by {assignment_clobber[0]}')
            new_students.append(new_student)
        return new_students
    return clobber_policy

# TODO Put this in another CSV or something.
COMMENTS = {
    12345678: {
        'Midterm': ['Hello world!'],
    },
}

def make_comments(comments: Dict[int, Dict[str, List[str]]]) -> Callable[[Student], List[Student]]:
    """Returns a policy function that adds comments.

    :param comments: A dict mapping student IDs to a dict mapping assignment names to the list of comments.
    :type comments: dict
    :returns: A comments policy function.
    :rtype: callable
    """
    def comments_policy(student: Student) -> List[Student]:
        if student.sid not in comments:
            return [student]
        new_student = copy.deepcopy(student)
        for assignment_name in comments[new_student.sid]:
            if assignment_name not in student.assignments:
                # If not present, it wasn't present in assignments CSV.
                raise RuntimeError(f'Comment references unknown assignment {assignment_name}')
            assignment = new_student.assignments[assignment_name]
            assignment_comments = comments[new_student.sid][assignment_name]
            assignment.grade.comments.extend(assignment_comments)
        return [new_student]
    return comments_policy

def generate_grade_reports(students: Dict[int, List[Student]]) -> Dict[int, GradeReport]:
    grade_reports: Dict[int, GradeReport] = {}
    for sid in students:
        for student in students[sid]:
            grade_report = student.get_grade_report()
            if sid not in grade_reports or grade_report.total_grade > grade_reports[sid].total_grade:
                grade_reports[sid] = grade_report
    return grade_reports

def dump_students(students: Dict[int, List[Student]], assignments: Dict[str, Assignment], categories: Dict[str, Category], rounding: Optional[int] = None, outfile: TextIO = sys.stdout) -> None:
    """Dumps students as a CSV to stdout.

    :param students: The students to dump.
    :type students: dict
    :param assignments: The assignments.
    :type assignments: dict
    :param categories: The categories.
    :type categories: dict
    :param rounding: The number of decimal places to round to, or None if no rounding.
    :type rounding: int
    """
    grade_reports = generate_grade_reports(students)

    # Derive output rows.
    header = ['SID', 'Name', 'Total Score', 'Percentile', 'Slip Days Used']
    for category in categories.values():
        header.append(f'Category: {category.name} - Raw Score')
        header.append(f'Category: {category.name} - Adjusted Score')
        header.append(f'Category: {category.name} - Weighted Score')
        header.append(f'Category: {category.name} - Comments')
    for assignment in assignments.values():
        header.append(f'{assignment.name} - Raw Score')
        header.append(f'{assignment.name} - Adjusted Score')
        header.append(f'{assignment.name} - Weighted Score')
        header.append(f'{assignment.name} - Comments')
    rows: List[List[Any]] = [header]
    for sid in students:
        grade_report = grade_reports[sid]
        row: List[Any] = [grade_report.student.sid, grade_report.student.name, grade_report.total_grade, 0.0, grade_report.student.slip_days_used] # 0.0 is temporary percentile.
        absent = ('no grades found', 'no grades found', 'no grades found', 'no grades found')
        for category in categories.values():
            if category.name in grade_report.categories:
                category_report = grade_report.categories[category.name]
                row.append(category_report.raw)
                row.append(category_report.adjusted)
                row.append(category_report.weighted)
                row.append(', '.join(category_report.comments))
            else:
                row.extend(absent)
        for assignment in assignments.values():
            if assignment.name in grade_report.assignments:
                assignment_report = grade_report.assignments[assignment.name]
                row.append(assignment_report.raw)
                row.append(assignment_report.adjusted)
                row.append(assignment_report.weighted)
                row.append(', '.join(assignment_report.comments))
            else:
                row.extend(absent)
        rows.append(row)

    # Compute percentiles.
    students_by_score = list(students.keys())
    students_by_score.sort(key=lambda sid: grade_reports[sid].total_grade, reverse=True)
    num_students = len(students)
    student_percentiles: Dict[int, float] = {}
    for rank in range(len(students)):
        sid = students_by_score[rank]
        student_percentiles[sid] = 1.0 - rank / num_students
    for row in rows:
        if row is header:
            continue
        sid = row[0]
        row[3] = student_percentiles[sid]

    # Round rows.
    if rounding is not None:
        for row in rows:
            for i in range(len(row)):
                if isinstance(row[i], float):
                    row[i] = round(row[i], rounding)

    csv.writer(outfile).writerows(rows)

def main(args: argparse.Namespace) -> None:
    roster_path: str = args.roster
    categories_path: str = args.categories
    assignments_path: str = args.assignments
    grades_path = args.grades
    overrides_path: Optional[str] = args.overrides
    clobbers_path: Optional[str] = args.clobbers
    extensions_path: Optional[str] = args.extensions
    accommodations_path: Optional[str] = args.accommodations
    output_path: Optional[str] = args.output
    rounding = int(args.rounding) if args.rounding else None

    categories = import_categories(categories_path)
    assignments = import_assignments(assignments_path, categories)
    overrides: Optional[Dict[int, Dict[str, float]]]
    if overrides_path is not None:
        overrides = import_overrides(overrides_path, assignments)
    else:
        overrides = None
    students = import_roster_and_grades(roster_path, grades_path, categories, assignments, overrides)

    if accommodations_path:
        students = apply_policy(make_accommodations(accommodations_path), students)
    if extensions_path:
        students = apply_policy(make_extensions(extensions_path), students)
    students = apply_policy(slip_day_policy, students)
    students = apply_policy(late_multiplier_policy, students)
    students = apply_policy(drop_policy, students)
    if clobbers_path:
        students = apply_policy(make_clobbers(clobbers_path, list(categories), list(assignments), students), students)
    students = apply_policy(make_comments(COMMENTS), students)

    if output_path is not None:
        with open(output_path, 'w') as outfile:
            dump_students(students, assignments, categories, rounding=rounding, outfile=outfile)
    else:
        dump_students(students, assignments, categories, rounding=rounding, outfile=sys.stdout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('roster', help='CSV roster downloaded from CalCentral')
    parser.add_argument('grades', help='CSV grades downloaded from Gradescope')
    parser.add_argument('categories', help='CSV with assignment categories')
    parser.add_argument('assignments', help='CSV with assignments')
    parser.add_argument('--overrides', '-d', help='CSV with score overrides')
    parser.add_argument('--clobbers', '-c', help='CSV with clobbers')
    parser.add_argument('--extensions', '-e', help='CSV with extensions')
    parser.add_argument('--accommodations', '-a', help='CSV with accommodations for drops and slip days')
    parser.add_argument('--rounding', '-r', help='Number of decimal places to round to')
    parser.add_argument('--output', '-o', help='Output CSV file')
    #parser.add_argument('--config', '--c', help='yaml file of configs')
    #parser.add_argument('-v', '--verbose', action='count', help='verbosity')
    args = parser.parse_args()
    main(args)
