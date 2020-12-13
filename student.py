import datetime
import enum
import statistics
from dataclasses import dataclass, field

from typing import Dict, List, NamedTuple, Optional, Tuple

@dataclass(frozen=True)
class Multiplier:
    multiplier: float
    description: str

@dataclass
class Category:
    name: str
    weight: float
    drops: int
    slip_days: int
    has_late_multiplier: bool
    override: Optional[float] = None
    comments: List[str] = field(default_factory=list)

@dataclass
class Assignment:
    @dataclass
    class Grade:
        score: float
        lateness: datetime.timedelta
        slip_days_applied: int = 0
        multipliers_applied: List[Multiplier] = field(default_factory=list)
        dropped: bool = False
        override: Optional[float] = None
        comments: List[str] = field(default_factory=list)

        def get_score(self) -> float:
            score = self.score if self.override is None else self.override
            for multiplier in self.multipliers_applied:
                score *= multiplier.multiplier
            return score

    name: str
    category: str
    score_possible: float
    weight: float
    slip_group: int
    _grade: Optional['Assignment.Grade'] = None

    @property
    def grade(self) -> 'Assignment.Grade':
        assert self._grade is not None, 'Grade is not yet initialized'
        return self._grade

    @grade.setter
    def grade(self, grade: 'Assignment.Grade') -> None:
        self._grade = grade

@dataclass
class GradeReport:
    @dataclass
    class CategoryEntry:
        raw: float
        adjusted: float
        weighted: float
        comments: List[str] = field(default_factory=list)

    @dataclass
    class AssignmentEntry:
        raw: float
        adjusted: float
        weighted: float
        comments: List[str] = field(default_factory=list)

    student: 'Student'
    total_grade: float = 0.0
    categories: Dict[str, 'GradeReport.CategoryEntry'] = field(default_factory=dict)
    assignments: Dict[str, 'GradeReport.AssignmentEntry'] = field(default_factory=dict)

@dataclass
class Student:
    sid: int
    name: str
    categories: Dict[str, Category]
    assignments: Dict[str, Assignment]
    slip_days_used: int = 0

    def get_grade_report(self) -> GradeReport:
        grade_report = GradeReport(self)
        for category in self.categories.values():
            assignments_in_category = list(assignment for assignment in self.assignments.values() if assignment.category == category.name)
            category_numerator = 0.0 # Category-weighted grades on assignments
            category_denominator = 0.0 # Total assignment weights

            # Category denominator.
            for assignment in assignments_in_category:
                grade = self.assignments[assignment.name].grade
                if not grade.dropped:
                    category_denominator += assignment.weight

            # AssignmentEntry objects with multipliers for adjusted score, weighted score, and cateogry numerator.
            for assignment in assignments_in_category:
                grade = self.assignments[assignment.name].grade
                assignment_raw_grade = grade.score / assignment.score_possible
                assignment_adjusted_grade = grade.get_score() / assignment.score_possible
                if not grade.dropped:
                    category_numerator += assignment_adjusted_grade * assignment.weight
                    assignment_weighted_grade = assignment_adjusted_grade / category_denominator * assignment.weight * category.weight
                else:
                    assignment_weighted_grade = 0.0
                assignment_comments = list(grade.comments)
                for multiplier in grade.multipliers_applied:
                    assignment_comments.append(f'x{multiplier.multiplier} ({multiplier.description})')
                grade_report.assignments[assignment.name] = GradeReport.AssignmentEntry(assignment_raw_grade, assignment_adjusted_grade, assignment_weighted_grade, assignment_comments)

            # CategoryEntry.
            category_raw_grade = category_numerator / category_denominator if category_denominator > 0.0 else 0.0
            if category.override is not None:
                category_adjusted_grade = category.override
            else:
                category_adjusted_grade = category_raw_grade
            category_weighted_grade = category_adjusted_grade * category.weight
            category_comments = list(category.comments)
            grade_report.categories[category.name] = GradeReport.CategoryEntry(category_raw_grade, category_adjusted_grade, category_weighted_grade, category_comments)

            # Add to total grade.
            grade_report.total_grade += category_weighted_grade

        return grade_report
