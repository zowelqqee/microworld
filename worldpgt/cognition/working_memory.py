"""Transient task working memory for cognitive answer sessions."""

from __future__ import annotations

from worldpgt.cognition.task_planner import build_task_plan
from worldpgt.cognition.types import (
    AnswerSelectionTrace,
    MemoryEvent,
    MemoryEventKind,
    ReasoningTrace,
    RepairKind,
    SessionTurnPlan,
    TaskMemorySnapshot,
    WorkingTask,
)


class TaskMemory:
    """In-process memory of the current reasoning task.

    This is not accepted memory, not an overlay, and not persisted. It only
    summarizes recent reasoning state so later turns can continue coherently.
    """

    def __init__(self) -> None:
        self.active_task: WorkingTask | None = None
        self.parked_tasks: list[WorkingTask] = []
        self.events: list[MemoryEvent] = []

    def record_reasoned_answer(
        self,
        *,
        question: str,
        answer_text: str,
        trace: ReasoningTrace | None,
        selection_trace: AnswerSelectionTrace | None = None,
    ) -> None:
        if trace is None:
            self._append_event(
                "answer_without_trace",
                subject=None,
                detail=question,
            )
            return

        task = _task_from_trace(
            trace=trace,
            answer_text=answer_text,
            selection_trace=selection_trace,
            previous=self.active_task,
        )
        if self.active_task is not None and self.active_task.subject != task.subject:
            self._park_active_task()
            task = _task_from_trace(
                trace=trace,
                answer_text=answer_text,
                selection_trace=selection_trace,
                previous=self._take_parked(task.subject),
            )
        elif self.active_task is None:
            previous = self._take_parked(task.subject)
            if previous is not None:
                task = _task_from_trace(
                    trace=trace,
                    answer_text=answer_text,
                    selection_trace=selection_trace,
                    previous=previous,
                )

        self.active_task = task
        self._append_event(
            "active_task_updated",
            subject=task.subject,
            detail=task.goal,
        )

    def record_session_plan(
        self,
        *,
        plan: SessionTurnPlan,
        answer_text: str,
    ) -> None:
        if plan.subject is None:
            return
        if self.active_task is not None and self.active_task.subject == plan.subject:
            self.active_task = WorkingTask(
                subject=self.active_task.subject,
                goal=self.active_task.goal,
                status="active",
                turn_count=self.active_task.turn_count,
                active_subgoals=self.active_task.active_subgoals,
                missing_roles=self.active_task.missing_roles,
                next_questions=plan.next_questions or self.active_task.next_questions,
                selected_candidate=self.active_task.selected_candidate,
                rejected_candidates=self.active_task.rejected_candidates,
                selected_surface_variant=self.active_task.selected_surface_variant,
                repair_kinds=self.active_task.repair_kinds,
                last_answer_text=answer_text,
            )
        self._append_event(
            "session_plan_used",
            subject=plan.subject,
            detail=plan.intent,
        )

    def known_subjects(self) -> tuple[str, ...]:
        subjects: list[str] = []
        if self.active_task is not None:
            subjects.append(self.active_task.subject)
        subjects.extend(task.subject for task in self.parked_tasks)
        return tuple(dict.fromkeys(subjects))

    def resume(self, subject: str) -> WorkingTask | None:
        if self.active_task is not None and _same_subject(self.active_task.subject, subject):
            self._append_event(
                "task_resumed",
                subject=self.active_task.subject,
                detail="already active",
            )
            return self.active_task

        parked = self._take_parked(subject)
        if parked is None:
            return None

        self._park_active_task()
        active = WorkingTask(
            subject=parked.subject,
            goal=parked.goal,
            status="active",
            turn_count=parked.turn_count,
            active_subgoals=parked.active_subgoals,
            missing_roles=parked.missing_roles,
            next_questions=parked.next_questions,
            selected_candidate=parked.selected_candidate,
            rejected_candidates=parked.rejected_candidates,
            selected_surface_variant=parked.selected_surface_variant,
            repair_kinds=parked.repair_kinds,
            last_answer_text=parked.last_answer_text,
        )
        self.active_task = active
        self._append_event("task_resumed", subject=active.subject, detail=active.goal)
        return active

    def inspect(self, subject: str) -> WorkingTask | None:
        task = self._task_for_subject(subject)
        if task is not None:
            self._append_event("task_inspected", subject=task.subject, detail=task.goal)
        return task

    def describe_open_tasks(self) -> str:
        self._append_event("memory_summarized", subject=None, detail="open_tasks")
        if self.active_task is None and not self.parked_tasks:
            return "There are no open reasoning tasks in this session."
        parts: list[str] = []
        if self.active_task is not None:
            parts.append(f"Active: {_describe_task(self.active_task)}")
        if self.parked_tasks:
            parked = "; ".join(_describe_task(task) for task in self.parked_tasks)
            parts.append(f"Parked: {parked}")
        return " ".join(parts)

    def describe_task(self, subject: str) -> str | None:
        task = self.inspect(subject)
        if task is None:
            return None
        return _describe_task(task)

    def summarize_active(self) -> str:
        self._append_event(
            "memory_summarized",
            subject=self.active_task.subject if self.active_task else None,
            detail="active_task",
        )
        if self.active_task is None:
            return "There is no active reasoning task in this session."
        return _summarize_active_task(self.active_task)

    def known_facts_summary(self) -> str:
        self._append_event(
            "memory_summarized",
            subject=self.active_task.subject if self.active_task else None,
            detail="known_supported",
        )
        if self.active_task is None:
            return "There is no active reasoning task to summarize."
        known = _known_answer_text(self.active_task.last_answer_text)
        if not known:
            return _describe_task(self.active_task)
        text = f"For {self.active_task.subject}, the supported answer so far is: {known}"
        missing = _join_list([str(role) for role in self.active_task.missing_roles])
        if missing:
            text += f" Still missing: {missing}."
        if self.active_task.next_questions:
            text += f" Next useful question: {self.active_task.next_questions[0]}"
        return _finish_sentence(text)

    def snapshot(self) -> TaskMemorySnapshot:
        return TaskMemorySnapshot(
            active_task=self.active_task,
            parked_tasks=tuple(self.parked_tasks),
            events=tuple(self.events),
        )

    def _park_active_task(self) -> None:
        if self.active_task is None:
            return
        parked = WorkingTask(
            subject=self.active_task.subject,
            goal=self.active_task.goal,
            status="parked",
            turn_count=self.active_task.turn_count,
            active_subgoals=self.active_task.active_subgoals,
            missing_roles=self.active_task.missing_roles,
            next_questions=self.active_task.next_questions,
            selected_candidate=self.active_task.selected_candidate,
            rejected_candidates=self.active_task.rejected_candidates,
            selected_surface_variant=self.active_task.selected_surface_variant,
            repair_kinds=self.active_task.repair_kinds,
            last_answer_text=self.active_task.last_answer_text,
        )
        self.parked_tasks = [
            task for task in self.parked_tasks if task.subject != parked.subject
        ]
        self.parked_tasks.append(parked)
        self._append_event("task_parked", subject=parked.subject, detail=parked.goal)

    def _take_parked(self, subject: str) -> WorkingTask | None:
        for idx, task in enumerate(self.parked_tasks):
            if _same_subject(task.subject, subject):
                return self.parked_tasks.pop(idx)
        return None

    def _task_for_subject(self, subject: str) -> WorkingTask | None:
        if self.active_task is not None and _same_subject(self.active_task.subject, subject):
            return self.active_task
        for task in self.parked_tasks:
            if _same_subject(task.subject, subject):
                return task
        return None

    def _append_event(
        self,
        kind: MemoryEventKind,
        *,
        subject: str | None,
        detail: str,
    ) -> None:
        self.events.append(
            MemoryEvent(
                index=len(self.events),
                kind=kind,
                subject=subject,
                detail=detail,
            )
        )


def _task_from_trace(
    *,
    trace: ReasoningTrace,
    answer_text: str,
    selection_trace: AnswerSelectionTrace | None,
    previous: WorkingTask | None,
) -> WorkingTask:
    task_plan = build_task_plan(trace)
    selected_evaluation = (
        selection_trace.evaluation_for(selection_trace.selected_name)
        if selection_trace is not None and selection_trace.selected_name is not None
        else None
    )
    repairs: tuple[RepairKind, ...] = ()
    if selected_evaluation is not None:
        repairs = tuple(repair.kind for repair in selected_evaluation.correction.repairs)
    rejected = ()
    if trace.deliberation is not None:
        rejected = tuple(
            evaluation.candidate_name
            for evaluation in trace.deliberation.evaluations
            if evaluation.decision == "rejected"
        )
    turn_count = previous.turn_count + 1 if previous is not None else 1
    return WorkingTask(
        subject=trace.task.subject,
        goal=task_plan.goal,
        status="active",
        turn_count=turn_count,
        active_subgoals=trace.working_memory.active_subgoals,
        missing_roles=trace.working_memory.missing_roles,
        next_questions=task_plan.next_questions,
        selected_candidate=(
            trace.deliberation.selected_name
            if trace.deliberation is not None
            else None
        ),
        rejected_candidates=rejected,
        selected_surface_variant=(
            selection_trace.selected_name if selection_trace is not None else None
        ),
        repair_kinds=repairs,
        last_answer_text=answer_text,
    )


def _same_subject(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def _describe_task(task: WorkingTask) -> str:
    status = task.status
    missing = _join_list([str(role) for role in task.missing_roles])
    subgoals = _join_list(list(task.active_subgoals))
    text = f"{task.subject} ({status}) — {task.goal}"
    if subgoals:
        text += f"; active subgoals: {subgoals}"
    if missing:
        text += f"; missing: {missing}"
    if task.next_questions:
        text += f"; next: {task.next_questions[0]}"
    return _finish_sentence(text)


def _summarize_active_task(task: WorkingTask) -> str:
    text = f"Active task: {_describe_task(task)}"
    known = _known_answer_text(task.last_answer_text)
    if known:
        text += f" Last supported answer: {known}"
    return _finish_sentence(text)


def _known_answer_text(answer_text: str) -> str:
    sentences = _sentences(answer_text)
    kept = [
        sentence
        for sentence in sentences
        if not sentence.lower().startswith("a useful next question would be:")
    ]
    return " ".join(kept).strip()


def _sentences(text: str) -> list[str]:
    import re

    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL_DOT>", text)
    parts = re.findall(r"[^.!?]+[.!?]|[^.!?]+$", protected)
    return [
        part.strip().replace("<DECIMAL_DOT>", ".")
        for part in parts
        if part.strip()
    ]


def _join_list(items: list[str]) -> str:
    items = [item for item in dict.fromkeys(items) if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _finish_sentence(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else f"{text}."
