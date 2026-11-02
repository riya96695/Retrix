import { useCallback, useEffect, useRef, useState } from "react";

// Until the answer arrives, stages advance one by one and wait at holdAt.
export default function useStageTicker(holdAt, intervalMs = 900) {
  const [active, setActive] = useState(-1);
  const activeRef = useRef(-1);
  const timer = useRef(null);

  const set = useCallback((v) => {
    activeRef.current = v;
    setActive(v);
  }, []);
  const stop = useCallback(() => {
    clearInterval(timer.current);
    timer.current = null;
  }, []);
  const start = useCallback(() => {
    stop();
    set(0);
    timer.current = setInterval(() => {
      if (activeRef.current < holdAt) set(activeRef.current + 1);
    }, intervalMs);
  }, [holdAt, intervalMs, set, stop]);
  const reset = useCallback(() => {
    stop();
    set(-1);
  }, [set, stop]);
  const getActive = useCallback(() => activeRef.current, []);

  useEffect(() => stop, [stop]);

  return { active, start, reset, getActive };
}
