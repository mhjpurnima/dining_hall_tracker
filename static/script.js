document.addEventListener("DOMContentLoaded", function () {
  // Fetch dining hall status
  const statusButton = document.getElementById("check-status");
  const statusElement = document.getElementById("status");
  const peakButton = document.getElementById("check-peak-hour");
  const peakElement = document.getElementById("peak-hour");

  const days = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
  const currentDay = days[new Date().getDay()]; // Gets current day abbreviation

  statusButton.addEventListener("click", function () {
    statusElement.innerText = "Checking...";
    fetch(`/get_status/${currentDay}`)
      .then((response) => response.json())
      .then((data) => {
        statusElement.innerText = data.status;
      })
      .catch((error) => {
        console.error("Error fetching status:", error);
        statusElement.innerText = "Error loading status";
      });
  });

  peakButton.addEventListener("click", function () {
    peakElement.innerText = "Checking...";
    fetch(`/get_peak_hour/${currentDay}`)
      .then((response) => response.json())
      .then((data) => {
        peakElement.innerText = data.peak_hours || "No data available";
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

  document
    .getElementById("check-quiet-hours")
    .addEventListener("click", function () {
      fetch("/get_quiet_hours")
        .then((response) => response.json())
        .then((data) => {
          const list = document.getElementById("quiet-hours-list");
          list.innerHTML = ""; // Clear previous results

          if (data.quiet_hours && data.quiet_hours.length > 0) {
            data.quiet_hours.forEach((hour) => {
              const li = document.createElement("li");
              li.innerHTML = `
                        ${hour.time} 
                        `;
              // <small>(Avg: ${hour.average_count} people,
              // Based on ${hour.readings_used} readings)</small>
              list.appendChild(li);
            });
          } else {
            list.innerHTML = "<li>No quiet hours data available</li>";
          }
        })
        .catch((error) => {
          console.error("Error:", error);
          document.getElementById("quiet-hours-list").innerHTML =
            "<li>Error loading quiet hours</li>";
        });
    });
});
