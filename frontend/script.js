const API = "http://127.0.0.1:5000/api";

let chart;

// ------------------------------
// Load Dashboard
// ------------------------------

document.addEventListener("DOMContentLoaded", () => {

  createChart();

  loadDashboard();

  setInterval(loadDashboard, 1000);

  const themeBtn = document.getElementById("themeBtn");

  function applyTheme(theme){

    if(theme==="dark"){

      document.body.classList.add("dark");
      themeBtn.innerText="Turn Light Mode On";

    }else{

      document.body.classList.remove("dark");
      themeBtn.innerText="Turn Dark Mode On";

    }

    localStorage.setItem("theme",theme);

  }

  const savedTheme = localStorage.getItem("theme") || "light";

  applyTheme(savedTheme);

  themeBtn.addEventListener("click",()=>{

    if(document.body.classList.contains("dark")){

      applyTheme("light");

    }else{

      applyTheme("dark");

    }

  });

});

// ------------------------------
// Main Loader
// ------------------------------

async function loadDashboard(){

  await Promise.all([
    getStatus(),
    getHistory()
  ]);

}

// ------------------------------
// STATUS
// ------------------------------

async function getStatus(){

  try{

    const response = await fetch(`${API}/status`);

    const data = await response.json();

    updateCards(data);

    updateFeatures(data.features);

  }

  catch(error){

    console.error(error);

  }

}

// ------------------------------
// HISTORY
// ------------------------------

async function getHistory(){

  try{

    const response = await fetch(`${API}/history`);

    const history = await response.json();

    updateChart(history);

  }

  catch(error){

    console.error(error);

  }

}

// ------------------------------
// SUMMARY CARDS
// ------------------------------

function updateCards(data){

  const status = document.getElementById("status");

  status.innerText = data.status;

  status.className = "";

  if(data.status === "ALERT"){

    status.classList.add("status-alert");

  }

  else if(data.status === "FATIGUED"){

    status.classList.add("status-fatigued");

  }

  else{

    status.classList.add("status-no_face_detected");

  }

  document.getElementById("probability").innerText =
      data.fatigue_probability == null
          ? "--"
          : (data.fatigue_probability*100).toFixed(1)+"%";

  const alarm = document.getElementById("alarm");

  alarm.innerText =
      data.alarm_triggered
          ? "ON"
          : "OFF";

  alarm.className = data.alarm_triggered
      ? "alarm-on"
      : "alarm-off";

  document.getElementById("ratio").innerText =
      (data.fatigued_ratio_in_window*100).toFixed(0)+"%";

  document.getElementById("timestamp").innerText =
      data.timestamp
          ? new Date(data.timestamp*1000).toLocaleTimeString()
          : "--";

}

// ------------------------------
// FEATURE CARDS
// ------------------------------

function updateFeatures(features){

  const grid = document.getElementById("featureGrid");

  grid.innerHTML = "";

  if(!features){

    return;

  }

  Object.entries(features).forEach(([key,value])=>{

    const card = document.createElement("div");

    card.className = "feature-card";

    card.innerHTML = `

            <h4>${formatTitle(key)}</h4>

            <p>${formatValue(value)}</p>

        `;

    grid.appendChild(card);

  });

}

// ------------------------------
// Create Chart
// ------------------------------

function createChart(){

  const ctx = document
      .getElementById("historyChart")
      .getContext("2d");

  chart = new Chart(ctx,{

    type:"line",

    data:{

      labels:[],

      datasets:[{

        label:"Fatigue Probability",

        data:[],

        borderColor:"#2563eb",

        backgroundColor:"rgba(37,99,235,.15)",

        borderWidth:3,

        tension:.4,

        fill:true

      }]

    },

    options:{

      responsive:true,

      plugins:{

        legend:{
          display:false
        }

      },

      scales:{

        y:{

          beginAtZero:true,

          max:1

        }

      }

    }

  });

}

// ------------------------------
// Update Chart
// ------------------------------

function updateChart(history){

  const labels = [];

  const values = [];

  history.forEach(item=>{

    labels.push(

        item.timestamp

            ? new Date(item.timestamp*1000).toLocaleTimeString()

            : "--"

    );

    values.push(

        item.fatigue_probability ?? 0

    );

  });

  chart.data.labels = labels;

  chart.data.datasets[0].data = values;

  chart.update();

}

// ------------------------------
// Helpers
// ------------------------------

function formatTitle(text){

  return text
      .replace(/_/g," ")
      .replace(/\b\w/g,c=>c.toUpperCase());

}

function formatValue(value){

  if(typeof value==="number"){

    return Number(value).toFixed(3);

  }

  return value;

}

