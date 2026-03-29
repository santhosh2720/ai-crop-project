const fieldConfig = [
  { name: "nitrogen", label: "Nitrogen" },
  { name: "phosphorous", label: "Phosphorous" },
  { name: "potassium", label: "Potassium" },
  { name: "ph", label: "pH" },
  { name: "temperature_c", label: "Temperature (C)" },
  { name: "humidity", label: "Humidity" },
  { name: "rainfall_mm", label: "Rainfall (mm)" },
  { name: "area", label: "Area (hectares)", readonly: true },
];

const INDIA_CENTER = [20.5937, 78.9629];
const MAP_ZOOM_DEFAULT = 5;
const MAP_ZOOM_CLOSE = 18;

const form = document.getElementById("prediction-form");
const formGrid = document.getElementById("form-grid");
const statusEl = document.getElementById("status");
const bestCropEl = document.getElementById("best-crop");
const table = document.getElementById("results-table");
const tbody = table.querySelector("tbody");
const metricsGrid = document.getElementById("training-metrics");
const areaDisplay = document.getElementById("area-display");
const coordsDisplay = document.getElementById("coords-display");
const recentRainfallDisplay = document.getElementById("recent-rainfall-display");
const climateRainfallDisplay = document.getElementById("climate-rainfall-display");
const locationBanner = document.getElementById("location-banner");
const locateBtn = document.getElementById("locate-btn");
const clearMapBtn = document.getElementById("clear-map-btn");
const drawPolygonBtn = document.getElementById("draw-polygon-btn");
const drawRectangleBtn = document.getElementById("draw-rectangle-btn");
const drawCircleBtn = document.getElementById("draw-circle-btn");

let metadata = null;
let map = null;
let drawnItems = null;
let activeShape = null;
let anchorMarker = null;
let drawHandlers = null;

const state = {
  areaHectares: null,
  latitude: null,
  longitude: null,
  locationLabel: "Mark or draw your land on the map.",
  recentRainfallMm: null,
  climateRainfallMm: null,
};

function setStatus(message) {
  statusEl.textContent = message;
}

function normalizeText(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function parseFlexibleNumber(value, fallback = null) {
  if (value === null || value === undefined) return fallback;
  const cleaned = String(value).trim().replace(/,/g, "");
  if (!cleaned) return fallback;
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return "--";
  return Number(value).toFixed(digits);
}

function createField(config, defaultValue) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  const label = document.createElement("label");
  label.setAttribute("for", config.name);
  label.textContent = config.label;

  const input = document.createElement("input");
  input.id = config.name;
  input.name = config.name;
  input.type = "text";
  input.inputMode = "decimal";
  if (defaultValue !== undefined && defaultValue !== null) {
    input.value = defaultValue;
  }
  if (config.readonly) {
    input.readOnly = true;
  }

  wrapper.append(label, input);
  return wrapper;
}

function renderForm() {
  const defaults = metadata?.default_inputs || {};
  formGrid.innerHTML = "";
  fieldConfig.forEach((config) => {
    const value = config.name === "area" ? "" : defaults[config.name];
    formGrid.appendChild(createField(config, value));
  });
}

function renderMetrics(report) {
  metricsGrid.innerHTML = "";
  if (!report) return;

  const cards = [
    ["Stacking Accuracy", report.classification_metrics?.stacking?.accuracy],
    ["Top-3 Accuracy", report.classification_metrics?.stacking?.top3_accuracy],
    ["LightGBM Accuracy", report.classification_metrics?.lightgbm?.accuracy],
    ["Yield R²", report.regression_metrics?.r2],
  ];

  cards.forEach(([label, value]) => {
    if (value === undefined) return;
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `<span>${label}</span><strong>${Number(value).toFixed(4)}</strong>`;
    metricsGrid.appendChild(card);
  });
}

function renderResults(result) {
  const best = result.top_crops[0];
  bestCropEl.classList.remove("hidden");
  bestCropEl.innerHTML = `
    <h3>Best Crop: ${best.crop}</h3>
    <p>
      Probability ${best.classification_probability.toFixed(4)} with yield
      ${best.predicted_yield.toFixed(2)}, profit ${best.profit.toFixed(2)},
      and final score ${best.final_score.toFixed(4)}.
    </p>
  `;

  tbody.innerHTML = "";
  result.top_crops.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.crop}</td>
      <td>${item.classification_probability.toFixed(4)}</td>
      <td>${item.predicted_yield.toFixed(2)}</td>
      <td>${item.profit.toFixed(2)}</td>
      <td>${item.risk.toFixed(4)}</td>
      <td>${item.sustainability_score.toFixed(4)}</td>
      <td>${item.final_score.toFixed(4)}</td>
    `;
    tbody.appendChild(row);
  });
  table.classList.remove("hidden");
  renderMetrics(result.training_summary);
}

function updateAreaDisplay() {
  if (!state.areaHectares) {
    areaDisplay.textContent = "Draw on map";
    return;
  }
  areaDisplay.textContent = `${formatNumber(state.areaHectares, 3)} ha`;
}

function updateCoordsDisplay() {
  if (!Number.isFinite(state.latitude) || !Number.isFinite(state.longitude)) {
    coordsDisplay.textContent = "Waiting for location";
    return;
  }
  coordsDisplay.textContent = `${formatNumber(state.latitude, 5)}, ${formatNumber(state.longitude, 5)}`;
}

function updateLocationBanner() {
  locationBanner.textContent = state.locationLabel;
}

function sumPositive(values) {
  return values
    .filter((value) => Number.isFinite(value) && Number(value) > 0)
    .reduce((sum, value) => sum + Number(value), 0);
}

function updateRainfallDisplays() {
  recentRainfallDisplay.textContent = Number.isFinite(state.recentRainfallMm)
    ? `${formatNumber(state.recentRainfallMm, 2)} mm`
    : "Waiting for map";
  climateRainfallDisplay.textContent = Number.isFinite(state.climateRainfallMm)
    ? `${formatNumber(state.climateRainfallMm, 2)} mm`
    : "Waiting for map";
}

async function fetchLiveWeather(lat, lng) {
  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lng}` +
    `&current=temperature_2m,relative_humidity_2m,rain` +
    `&forecast_days=1&timezone=auto`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Weather API request failed.");
  }
  const data = await response.json();
  const current = data.current || {};
  return {
    temperature: current.temperature_2m,
    humidity: current.relative_humidity_2m,
    currentRain: current.rain,
  };
}

function formatDate(date) {
  return date.toISOString().slice(0, 10);
}

async function fetchRainfallHistory(lat, lng) {
  const endDate = new Date();
  const startDate = new Date();
  startDate.setDate(endDate.getDate() - 364);

  const url =
    `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lng}` +
    `&start_date=${formatDate(startDate)}&end_date=${formatDate(endDate)}` +
    `&daily=precipitation_sum&timezone=auto`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Rainfall history request failed.");
  }
  const data = await response.json();
  const daily = data.daily || {};
  const precipitationSeries = Array.isArray(daily.precipitation_sum) ? daily.precipitation_sum.map(Number) : [];

  const annualTotal = sumPositive(precipitationSeries);
  const recent30Total = sumPositive(precipitationSeries.slice(-30));
  const climateMonthlyEquivalent = annualTotal > 0 ? annualTotal / 12 : null;

  return {
    annualTotal,
    recent30Total,
    climateMonthlyEquivalent,
  };
}

async function fetchWeather(lat, lng) {
  const [liveWeather, history] = await Promise.all([
    fetchLiveWeather(lat, lng),
    fetchRainfallHistory(lat, lng),
  ]);

  const temperature = liveWeather.temperature;
  const humidity = liveWeather.humidity;
  const recent30Total = history.recent30Total;
  const climateMonthlyEquivalent = history.climateMonthlyEquivalent;
  const currentRain = liveWeather.currentRain;

  let modelRainfall = null;
  if (Number.isFinite(climateMonthlyEquivalent) && Number.isFinite(recent30Total)) {
    modelRainfall = 0.7 * climateMonthlyEquivalent + 0.3 * recent30Total;
  } else if (Number.isFinite(climateMonthlyEquivalent)) {
    modelRainfall = climateMonthlyEquivalent;
  } else if (Number.isFinite(recent30Total)) {
    modelRainfall = recent30Total;
  } else if (Number.isFinite(currentRain)) {
    modelRainfall = currentRain;
  }

  state.recentRainfallMm = Number.isFinite(recent30Total) ? recent30Total : currentRain;
  state.climateRainfallMm = Number.isFinite(modelRainfall) ? modelRainfall : null;
  updateRainfallDisplays();

  if (temperature !== undefined) document.getElementById("temperature_c").value = temperature;
  if (humidity !== undefined) document.getElementById("humidity").value = humidity;
  if (modelRainfall !== undefined && modelRainfall !== null) {
    document.getElementById("rainfall_mm").value = formatNumber(modelRainfall, 2);
  }
}

function setAnchorLocation(lat, lng) {
  state.latitude = lat;
  state.longitude = lng;
  updateCoordsDisplay();

  if (!anchorMarker) {
    anchorMarker = L.marker([lat, lng]).addTo(map);
  } else {
    anchorMarker.setLatLng([lat, lng]);
  }
}

async function syncLocationAndWeather(lat, lng) {
  setAnchorLocation(lat, lng);
  setStatus("Fetching location and live weather for the selected land...");
  state.locationLabel = `Selected location: ${formatNumber(lat, 5)}, ${formatNumber(lng, 5)}`;
  updateLocationBanner();
  await fetchWeather(lat, lng).catch(() => {
    setStatus("Weather fetch failed. You can still type values manually.");
  });
  setStatus("Land location synced. Draw the farm boundary, then run optimization.");
}

function clearActiveShape() {
  if (activeShape && drawnItems) {
    drawnItems.removeLayer(activeShape);
  }
  activeShape = null;
  state.areaHectares = null;
  updateAreaDisplay();
  const areaInput = document.getElementById("area");
  if (areaInput) {
    areaInput.value = "";
  }
  state.recentRainfallMm = null;
  state.climateRainfallMm = null;
  updateRainfallDisplays();
}

function updateAreaFromLayer(layer) {
  let squareMeters = null;
  if (layer instanceof L.Circle) {
    squareMeters = Math.PI * layer.getRadius() * layer.getRadius();
  } else if (window.L && L.GeometryUtil && typeof L.GeometryUtil.geodesicArea === "function") {
    const latLngGroups = layer.getLatLngs();
    const points = Array.isArray(latLngGroups[0]) ? latLngGroups[0] : latLngGroups;
    squareMeters = L.GeometryUtil.geodesicArea(points);
  }

  if (Number.isFinite(squareMeters) && squareMeters > 0) {
    state.areaHectares = squareMeters / 10000;
  } else {
    state.areaHectares = null;
  }
  updateAreaDisplay();
  const areaInput = document.getElementById("area");
  if (areaInput) {
    areaInput.value = state.areaHectares ? formatNumber(state.areaHectares, 4) : "";
  }
}

function getLayerCenter(layer) {
  if (typeof layer.getBounds === "function" && layer.getBounds().isValid()) {
    return layer.getBounds().getCenter();
  }
  if (typeof layer.getLatLng === "function") {
    return layer.getLatLng();
  }
  return null;
}

function handleShapeChange(layer) {
  activeShape = layer;
  updateAreaFromLayer(layer);
  const center = getLayerCenter(layer);
  if (center) {
    if (typeof layer.getBounds === "function") {
      map.fitBounds(layer.getBounds(), { maxZoom: MAP_ZOOM_CLOSE, padding: [30, 30] });
    } else {
      map.setView(center, MAP_ZOOM_CLOSE);
    }
    syncLocationAndWeather(center.lat, center.lng).catch((error) => setStatus(error.message));
  }
}

function initMap() {
  map = L.map("map", {
    zoomControl: false,
    preferCanvas: true,
  }).setView(INDIA_CENTER, MAP_ZOOM_DEFAULT);

  window.setTimeout(() => {
    map.invalidateSize();
  }, 150);

  L.control.zoom({ position: "topright" }).addTo(map);

  L.esri.basemapLayer("Imagery").addTo(map);
  L.esri.basemapLayer("ImageryLabels").addTo(map);

  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  const drawControl = new L.Control.Draw({
    position: "topright",
    draw: {
      polygon: { allowIntersection: false, showArea: true },
      rectangle: true,
      circle: true,
      marker: false,
      polyline: false,
      circlemarker: false,
    },
    edit: {
      featureGroup: drawnItems,
      edit: true,
      remove: true,
    },
  });
  map.addControl(drawControl);

  drawHandlers = {
    polygon: new L.Draw.Polygon(map, drawControl.options.draw.polygon),
    rectangle: new L.Draw.Rectangle(map, drawControl.options.draw.rectangle),
    circle: new L.Draw.Circle(map, drawControl.options.draw.circle),
  };

  map.on(L.Draw.Event.CREATED, (event) => {
    clearActiveShape();
    const layer = event.layer;
    drawnItems.addLayer(layer);
    handleShapeChange(layer);
  });

  map.on(L.Draw.Event.EDITED, (event) => {
    event.layers.eachLayer((layer) => handleShapeChange(layer));
  });

  map.on(L.Draw.Event.DELETED, () => {
    activeShape = null;
    state.areaHectares = null;
    state.recentRainfallMm = null;
    state.climateRainfallMm = null;
    updateAreaDisplay();
    updateRainfallDisplays();
    const areaInput = document.getElementById("area");
    if (areaInput) {
      areaInput.value = "";
    }
  });

  map.on("click", (event) => {
    syncLocationAndWeather(event.latlng.lat, event.latlng.lng).catch((error) => setStatus(error.message));
  });
}

function enableDrawMode(mode) {
  if (!drawHandlers || !drawHandlers[mode]) {
    setStatus("Drawing tools are not ready yet. Refresh once and try again.");
    return;
  }
  drawHandlers[mode].enable();
  setStatus(`Drawing mode active: ${mode}. Mark your land boundary on the map.`);
}

function collectPayload() {
  const payload = { top_k: 3 };

  fieldConfig.forEach((field) => {
    payload[field.name] = parseFlexibleNumber(document.getElementById(field.name)?.value);
  });

  return payload;
}

async function loadMetadata() {
  const response = await fetch("/api/metadata");
  if (!response.ok) {
    throw new Error("Models are not trained yet. Train the backend first.");
  }
  metadata = await response.json();
  renderForm();
  renderMetrics(metadata.training_report);
  setStatus(
    `Loaded ${metadata.crop_count} real crops with ${metadata.recommendation_rows} agronomy rows and ${metadata.production_rows} production rows.`
  );
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    if (!state.areaHectares) {
      throw new Error("Draw your land boundary on the map first so area can be calculated.");
    }

    setStatus("Running crop prediction on the selected land...");
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed");
    }
    renderResults(data);
    setStatus(`Prediction completed. Best crop: ${data.best_crop}`);
  } catch (error) {
    setStatus(error.message);
  }
});

locateBtn.addEventListener("click", () => {
  if (!navigator.geolocation) {
    setStatus("Geolocation is not available in this browser.");
    return;
  }

  setStatus("Fetching your device location...");
  navigator.geolocation.getCurrentPosition(
    (position) => {
      const { latitude, longitude } = position.coords;
      map.setView([latitude, longitude], 17);
      syncLocationAndWeather(latitude, longitude).catch((error) => setStatus(error.message));
    },
    () => setStatus("Unable to access your current location."),
    { enableHighAccuracy: true, timeout: 15000 }
  );
});

clearMapBtn.addEventListener("click", () => {
  clearActiveShape();
  updateAreaDisplay();
  setStatus("Map drawing cleared. Click or draw again to continue.");
});

drawPolygonBtn.addEventListener("click", () => enableDrawMode("polygon"));
drawRectangleBtn.addEventListener("click", () => enableDrawMode("rectangle"));
drawCircleBtn.addEventListener("click", () => enableDrawMode("circle"));

initMap();
updateRainfallDisplays();
loadMetadata().catch((error) => {
  renderForm();
  setStatus(error.message);
});
