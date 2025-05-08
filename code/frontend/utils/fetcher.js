// frontend/utils/fetcher.js
export const fetcher = async (...args) => {
  const res = await fetch(...args);

  // If the status code is not in the range 200-299,
  // we still try to parse and throw it.
  if (!res.ok) {
    const error = new Error('An error occurred while fetching the data.');
    // Attempt to get error details from the response body
    try {
      error.info = await res.json(); // Get JSON error response if available
    } catch (e) {
      error.info = res.statusText; // Fallback to status text
    }
    error.status = res.status;
    console.error("API Fetch Error:", error);
    throw error;
  }

  return res.json();
};