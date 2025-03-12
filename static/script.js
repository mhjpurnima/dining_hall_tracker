document.addEventListener("DOMContentLoaded", function () {
  // Fetch dining hall status
  const statusButton = document.getElementById("check-status");
  const statusElement = document.getElementById("status");
  const peakButton = document.getElementById("check-peak-hour");
  const peakElement = document.getElementById("peak-hour");

  statusButton.addEventListener("click", function () {
    statusElement.innerText = "Checking...";
    fetch("/get_status")
      .then((response) => response.json())
      .then((data) => {
        statusElement.innerText = data.status;
      })
      .catch((error) => {
        console.error("Error fetching status:", error);
        statusElement.innerText = "Error checking status";
      });
  });

  peakButton.addEventListener("click", function () {
    peakElement.innerText = "Checking...";
    fetch("/get_peak_hour")
      .then((response) => response.json())
      .then((data) => {
        peakElement.innerText = data.peak_hour || "No data available";
      })
      .catch((error) => {
        console.error("Error fetching peak hour:", error);
        peakElement.innerText = "Error checking peak hours";
      });
  });

  // Fetch busiest hour
  // fetch("/get_busiest_hour")
  //   .then((response) => response.json())
  //   .then((data) => {
  //     document.getElementById("busiest-hour").innerText = data.busiest_hour;
  //   })
  //   .catch((error) => console.error("Error fetching busiest hour:", error));

  // // Fetch quiet hours
  // fetch("/get_quiet_hours")
  //   .then((response) => response.json())
  //   .then((data) => {
  //     document.getElementById("quiet-hours").innerText =
  //       data.quiet_hours.join(", ");
  //   })
  //   .catch((error) => console.error("Error fetching quiet hours:", error));
});
