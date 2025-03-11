document.addEventListener("DOMContentLoaded", function () {
    // Fetch dining hall status
    fetch("/get_status")
        .then(response => response.json())
        .then(data => {
            document.getElementById("status").innerText = data.status;
        })
        .catch(error => console.error("Error fetching status:", error));

    // Fetch busiest hour
    fetch("/get_busiest_hour")
        .then(response => response.json())
        .then(data => {
            document.getElementById("busiest-hour").innerText = data.busiest_hour;
        })
        .catch(error => console.error("Error fetching busiest hour:", error));

    // Fetch quiet hours
    fetch("/get_quiet_hours")
        .then(response => response.json())
        .then(data => {
            document.getElementById("quiet-hours").innerText = data.quiet_hours.join(", ");
        })
        .catch(error => console.error("Error fetching quiet hours:", error));
});
