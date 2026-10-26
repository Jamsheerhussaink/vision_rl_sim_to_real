"""Process-local cumulative lessons; preserve the original experiment on disk."""
LESSONS = ('approach', 'grasp', 'lift', 'place')


def activate(lesson):
    import sys
    from . import stages
    if lesson not in LESSONS:
        raise ValueError(f'Unknown lesson: {lesson}')
    terminal = LESSONS.index(lesson)
    original = stages.StageMachine

    class LessonStageMachine(original):
        def __init__(self, *args, **kwargs):
            # Lesson boundaries below replace the full-task terminal goal.
            kwargs['goal'] = 'place'
            super().__init__(*args, **kwargs)

        def update(self, measurements, dt, inspection=False):
            reward = super().update(measurements, dt, inspection=inspection)
            passed = (self.stage > terminal) & ~self.failure
            self.success |= passed
            self.stage.clamp_(max=terminal)
            return reward

    stages.StageMachine = LessonStageMachine
    stages.VERSION = f'simple_curriculum_v1_{lesson}'
    sys.modules['isaac_vision_rl.utils.four_stage'] = stages
