import asyncio
import time
from collections import deque
from typing import List, Tuple

class AsyncAnthropicRateLimiter:
    """
    Manages token usage for Anthropic API calls to respect rate limits.
    Tracks tokens used over a sliding time window.
    """
    def __init__(self, max_tokens_per_window: int, time_window_seconds: int = 60, check_interval_seconds: float = 0.5):
        """
        Initializes the rate limiter.

        Args:
            max_tokens_per_window: Maximum number of tokens allowed in the time window.
            time_window_seconds: Duration of the time window in seconds (e.g., 60 for per minute).
            check_interval_seconds: How often to check if the window has cleared when waiting.
        """
        self.max_tokens_per_window = max_tokens_per_window
        self.time_window_seconds = time_window_seconds
        self.check_interval_seconds = check_interval_seconds
        
        self._lock = asyncio.Lock()
        self._token_usage_timestamps: deque[Tuple[float, int]] = deque() # Stores (timestamp, tokens_used)

    def _cleanup_old_tokens(self) -> None:
        """Removes token usage records older than the current time window."""
        current_time = time.monotonic()
        while self._token_usage_timestamps and self._token_usage_timestamps[0][0] < current_time - self.time_window_seconds:
            self._token_usage_timestamps.popleft()

    def _current_tokens_in_window(self) -> int:
        """Calculates the total tokens used in the current time window after cleanup."""
        self._cleanup_old_tokens()
        return sum(tokens for _, tokens in self._token_usage_timestamps)

    async def acquire(self, tokens_needed: int) -> None:
        """
        Acquires permission to use a certain number of tokens.
        If the request would exceed the rate limit, it waits until capacity is available.
        """
        if tokens_needed > self.max_tokens_per_window:
            # This single request is larger than the entire window capacity.
            # Depending on strictness, could raise an error or just log a warning and proceed
            # assuming the API might handle it or it's a very bursty low-frequency large request.
            # For now, let's print a warning.
            print(f"[WARN] AnthropicRateLimiter: Requested tokens ({tokens_needed}) exceed max_tokens_per_window ({self.max_tokens_per_window}).")
            # No waiting logic here for this specific case, as waiting for window to clear won't help if one request is too big.

        async with self._lock:
            while True:
                current_tokens = self._current_tokens_in_window()
                if current_tokens + tokens_needed <= self.max_tokens_per_window:
                    self._token_usage_timestamps.append((time.monotonic(), tokens_needed))
                    # print(f"[DEBUG] AnthropicRateLimiter: Acquired {tokens_needed}. Current window usage: {current_tokens + tokens_needed}/{self.max_tokens_per_window}")
                    break
                else:
                    # Calculate how long to wait.
                    # A simple approach: wait for the check_interval.
                    # A more complex one: estimate when the oldest tokens will expire.
                    wait_time = self.check_interval_seconds
                    
                    # If there are tokens in the queue, we might be able to wait until the oldest one expires
                    if self._token_usage_timestamps:
                        oldest_token_timestamp = self._token_usage_timestamps[0][0]
                        time_to_oldest_expiry = (oldest_token_timestamp + self.time_window_seconds) - time.monotonic()
                        wait_time = max(0, min(self.check_interval_seconds, time_to_oldest_expiry)) # Wait shorter of check_interval or until oldest expires
                        if wait_time < 0.001: # Avoid near-zero sleeps if expiry is imminent
                            wait_time = 0.001 


                    # print(f"[DEBUG] AnthropicRateLimiter: Limit would be exceeded ({current_tokens + tokens_needed}/{self.max_tokens_per_window}). Waiting for {wait_time:.2f}s.")
                    await asyncio.sleep(wait_time)
                    # Loop continues, _cleanup_old_tokens and _current_tokens_in_window will be re-evaluated 