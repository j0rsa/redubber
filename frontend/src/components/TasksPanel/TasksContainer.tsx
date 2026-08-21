import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useActiveTasks } from '../../hooks/useActiveTasks';
import type { TaskStatus } from '../../types';
import { writeDebugLog } from '../../utils/debugLog';
import { TasksIndicator } from './TasksPanel';
import { TasksPanel } from './TasksPanel';

export const projectGlobalTasks = (tasks: TaskStatus[]) => {
  const visibleTasks = tasks.filter((task) => task.task_type !== 'reset_dub');
  const activeCount = visibleTasks.filter(
    (task) => task.status === 'queued' || task.status === 'running',
  ).length;

  return {
    activeTasks: visibleTasks,
    activeCount,
    hasActive: activeCount > 0,
  };
};

/**
 * Connected component that wires useActiveTasks data to the TasksIndicator
 * and TasksPanel UI. Must be rendered inside BrowserRouter so useNavigate works.
 */
export const TasksContainer = () => {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();
  const taskState = useActiveTasks();
  const { activeTasks, activeCount, hasActive } = projectGlobalTasks(taskState.activeTasks);

  // #region agent log
  useEffect(() => {
    writeDebugLog({
      hypothesisId: 'C',
      location: 'TasksContainer.tsx:activeTasks',
      message: 'Global task UI projection',
      data: {
        activeCount,
        hasActive,
        taskIds: activeTasks.map((task) => task.task_id),
        panelOpen: isOpen,
      },
    });
  }, [activeCount, activeTasks, hasActive, isOpen]);
  // #endregion

  // #region agent log
  useEffect(() => {
    writeDebugLog({
      hypothesisId: 'C',
      location: 'TasksContainer.tsx:globalProjection',
      message: 'Filtered global task UI projection',
      data: {
        sourceTaskIds: taskState.activeTasks.map((task) => task.task_id),
        visibleTaskIds: activeTasks.map((task) => task.task_id),
        activeCount,
        hasActive,
      },
    });
  }, [activeCount, activeTasks, hasActive, taskState.activeTasks]);
  // #endregion

  const handleViewJob = (taskId: string) => {
    setIsOpen(false);
    navigate(`/job/${taskId}`);
  };

  if (!hasActive) return null;

  return (
    <>
      <TasksIndicator
        activeCount={activeCount}
        isOpen={isOpen}
        onClick={() => setIsOpen((prev) => !prev)}
      />
      <TasksPanel
        tasks={activeTasks}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        onViewJob={handleViewJob}
      />
    </>
  );
};
