{% extends "base.html" %}

{% block titulo %}Prioridad de abastecimiento{% endblock %}
{% block titulo_pagina %}Prioridad de abastecimiento{% endblock %}
{% block subtitulo_pagina %}Qué productos vigilar hoy · Modelo independiente (RandomForest){% endblock %}

{% block topbar_acciones %}
<select id="selectorFecha" class="date-select"></select>
<a href="{{ url_for('upload.index') }}" class="btn-upload">
  <i data-lucide="upload"></i> Subir datos
</a>
{% endblock %}

{% block estilos %}
<style>
  .date-select { border: 1px solid var(--border); border-radius: 10px; padding: .55rem .9rem;
                 font-size: .85rem; font-weight: 600; background: var(--white); color: var(--dark); }

  .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.25rem; }
  .stat-box { border-radius: 14px; padding: 1.1rem 1.25rem; border: 1px solid var(--border); background: var(--white); }
  .stat-box.alta  { background: #FEF2F2; border-color: #FECACA; }
  .stat-box.media { background: #FFFBEB; border-color: #FDE68A; }
  .stat-box.baja  { background: #F0FDF4; border-color: #BBF7D0; }
  .stat-label { font-size: .82rem; font-weight: 700; margin-bottom: .3rem; }
  .stat-box.alta .stat-label  { color: #B91C1C; }
  .stat-box.media .stat-label { color: #92400E; }
  .stat-box.baja .stat-label  { color: #15803D; }
  .stat-num { font-size: 1.9rem; font-weight: 800; color: var(--dark); }
  .stat-sub { font-size: .78rem; color: var(--gray-mid); margin-top: .2rem; }

  .abs-layout { display: grid; grid-template-columns: 2fr 1fr; gap: 1rem; align-items: start; }
  .abs-card { background: var(--white); border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem; }

  .toolbar { display: flex; gap: .6rem; margin: 1rem 0; flex-wrap: wrap; }
  .toolbar input, .toolbar select { border: 1px solid var(--border); border-radius: 9px;
    padding: .5rem .75rem; font-size: .84rem; }
  .toolbar input { flex: 1; min-width: 180px; }

  table.abs-tabla { width: 100%; border-collapse: collapse; font-size: .85rem; }
  .abs-tabla th { text-align: left; color: var(--gray-mid); font-weight: 600; font-size: .72rem;
                  text-transform: uppercase; padding: .5rem .5rem; border-bottom: 1px solid var(--border); }
  .abs-tabla td { padding: .6rem .5rem; border-bottom: 1px solid var(--border); vertical-align: middle; }

  .badge { display: inline-flex; padding: .28rem .65rem; border-radius: 999px; font-size: .76rem; font-weight: 700; }
  .badge-alta  { background: #FEE2E2; color: #B91C1C; }
  .badge-media { background: #FEF3C7; color: #92400E; }
  .badge-baja  { background: #DCFCE7; color: #15803D; }

  .conf-bar-bg { width: 70px; height: 6px; border-radius: 4px; background: var(--gray-lt); overflow: hidden; }
  .conf-bar-fill { height: 100%; border-radius: 4px; }
  .conf-pct { font-size: .78rem; color: var(--gray-mid); margin-left: 6px; }

  .chip { display: inline-block; padding: .2rem .55rem; border-radius: 7px; font-size: .72rem;
          font-weight: 600; background: #FEE2E2; color: #991B1B; margin: 2px 3px 2px 0; }

  .paginacion { display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;
                font-size: .82rem; color: var(--gray-mid); }
  .pag-botones { display: flex; gap: .3rem; }
  .pag-btn { width: 30px; height: 30px; border-radius: 8px; border: 1px solid var(--border);
             background: var(--white); cursor: pointer; font-size: .8rem; }
  .pag-btn.active { background: var(--purple); color: #fff; border-color: var(--purple); }

  .metric-line { display: flex; justify-content: space-between; align-items: center;
                 font-size: .84rem; padding: .45rem 0; border-bottom: 1px dashed var(--border); }
  .metric-line:last-child { border-bottom: none; }
  .delta { font-size: .74rem; font-weight: 700; margin-left: 6px; }
  .delta.up { color: #15803D; } .delta.down { color: #B91C1C; }

  .aviso { background: #EFF6FF; color: #1E40AF; padding: .8rem 1rem; border-radius: 10px;
           font-size: .85rem; margin-bottom: 1rem; }

  .btn-secundario { border: 1px solid var(--border); background: var(--white); border-radius: 9px;
                     padding: .5rem .9rem; font-size: .82rem; font-weight: 600; cursor: pointer; }

  .modal-fondo { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.4);
                 align-items: center; justify-content: center; z-index: 50; }
  .modal-fondo.abierto { display: flex; }
  .modal-caja { background: var(--white); border-radius: 14px; padding: 1.5rem; max-width: 520px;
                width: 90%; max-height: 85vh; overflow-y: auto; }
  .modal-cerrar { float: right; cursor: pointer; border: none; background: none; font-size: 1.1rem; }
</style>
{% endblock %}

{% block contenido %}
<div class="aviso">
  Este modelo no predice demanda. Identifica qué productos requieren más atención para planificar tu abastecimiento.
</div>

<div id="estadoVacio" style="display:none;" class="abs-card"><p id="mensajeError"></p></div>

<div id="contenidoPrincipal" style="display:none;">

  <div class="stats-row">
    <div class="stat-box alta">
      <div class="stat-label">Alta prioridad</div>
      <div class="stat-num" id="numAlta">0</div>
      <div class="stat-sub" id="pctAlta">0% del total</div>
    </div>
    <div class="stat-box media">
      <div class="stat-label">Media prioridad</div>
      <div class="stat-num" id="numMedia">0</div>
      <div class="stat-sub" id="pctMedia">0% del total</div>
    </div>
    <div class="stat-box baja">
      <div class="stat-label">Baja prioridad</div>
      <div class="stat-num" id="numBaja">0</div>
      <div class="stat-sub" id="pctBaja">0% del total</div>
    </div>
    <div class="stat-box">
      <div class="stat-label" style="color:var(--gray-mid);">Productos totales</div>
      <div class="stat-num" id="numTotal">0</div>
      <div class="stat-sub">Activos para hoy</div>
    </div>
  </div>

  <div class="abs-layout">
    <div class="abs-card">
      <h3 style="margin:0 0 .8rem;">Productos por prioridad</h3>
      <div class="toolbar">
        <input type="text" id="buscar" placeholder="Buscar producto...">
        <select id="filtroPrioridad">
          <option value="">Todas las prioridades</option>
          <option value="Alta">Alta</option>
          <option value="Media">Media</option>
          <option value="Baja">Baja</option>
        </select>
        <select id="filtroCategoria"><option value="">Todas las categorías</option></select>
      </div>

      <table class="abs-tabla">
        <thead>
          <tr><th>Producto</th><th>Categoría</th><th>Prioridad</th><th>Confianza</th><th>Factores clave</th></tr>
        </thead>
        <tbody id="tablaProductos"></tbody>
      </table>

      <div class="paginacion">
        <span id="resumenPaginacion"></span>
        <div class="pag-botones" id="pagBotones"></div>
      </div>
    </div>

    <div>
      <div class="abs-card" style="margin-bottom:1rem;">
        <h3 style="margin-top:0;">Distribución por prioridad</h3>
        <canvas id="donutPrioridad" height="180"></canvas>
      </div>

      <div class="abs-card" style="margin-bottom:1rem;">
        <h3 style="margin-top:0;">Calidad del modelo</h3>
        <div class="metric-line"><span>Accuracy</span>
          <strong id="mAcc">—<span class="delta" id="dAcc"></span></strong></div>
        <div class="metric-line"><span>F1 score (macro)</span>
          <strong id="mF1">—<span class="delta" id="dF1"></span></strong></div>
        <button class="btn-secundario" style="margin-top:.8rem;width:100%;" onclick="abrirMatriz()">
          Ver matriz de confusión
        </button>
      </div>

      <div class="abs-card">
        <h3 style="margin-top:0;">Sobre este modelo</h3>
        <p style="font-size:.83rem;color:var(--gray-mid);">
          Clasifica cada producto por día en Alta, Media o Baja prioridad de abastecimiento.
        </p>
        <a href="#" class="btn-secundario" style="display:inline-block;text-decoration:none;"
           onclick="document.getElementById('modalMetodologia').classList.add('abierto');return false;">
          Ver metodología
        </a>
      </div>
    </div>
  </div>

  <div style="text-align:right;margin-top:1rem;">
    <button class="btn-secundario" onclick="reentrenar()">Reentrenar modelo</button>
  </div>
</div>

<!-- Modal: matriz de confusión -->
<div class="modal-fondo" id="modalMatriz">
  <div class="modal-caja">
    <button class="modal-cerrar" onclick="document.getElementById('modalMatriz').classList.remove('abierto')">✕</button>
    <h3 style="margin-top:0;">Matriz de confusión (holdout)</h3>
    <p style="font-size:.8rem;color:var(--gray-mid);">
      Cada grupo de barras es la clase REAL; la altura muestra cuántos productos-día se predijeron
      en cada clase. La diagonal (misma clase real y predicha) son los aciertos.
    </p>
    <img id="matrizImg" src="" alt="Matriz de confusión" style="max-width:100%;display:block;margin:0 auto;" />
    <div class="matriz-leyenda">
      <span>Fila = clase real · el % es dentro de esa fila (recall)</span>
      <span>Diagonal más oscura = aciertos</span>
    </div>
  </div>
</div>

<!-- Modal: metodología -->
<div class="modal-fondo" id="modalMetodologia">
  <div class="modal-caja">
    <button class="modal-cerrar" onclick="document.getElementById('modalMetodologia').classList.remove('abierto')">✕</button>
    <h3 style="margin-top:0;">¿Cómo se calcula la prioridad?</h3>
    <p style="font-size:.83rem;color:var(--dark);">
      Por cada producto se miden, sobre su historial real: <strong>rotación</strong> (volumen típico),
      <strong>volatilidad</strong> (qué tan errática es su demanda) e <strong>impacto de promoción</strong>.
      Cada una se lleva a un percentil de 0 a 1 respecto a los demás productos — sin pesos manuales.
      Se combinan con la <strong>presión del día</strong> (fin de semana, feriado, quincena, puente)
      para dar un puntaje de criticidad, que se corta en terciles: Baja / Media / Alta.
    </p>
    <p style="font-size:.83rem;color:var(--dark);">
      El modelo (Random Forest) aprende a predecir esa etiqueta a partir de la <strong>categoría</strong>
      del producto y el contexto del día — nunca del precio ni del histórico de demanda directamente —
      para poder generalizar a productos nuevos sin historial propio.
    </p>
  </div>
</div>
{% endblock %}

{% block scripts %}
<script>
let datosDias = [];
let categoriasDisponibles = [];
let productosFiltrados = [];
let paginaActual = 1;
const POR_PAGINA = 9;
let chartDonut = null;

async function cargar() {
  try {
    const res = await fetch("/abastecimiento/ejecutar", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({dias: 7}),
    });
    const data = await res.json();
    if (!data.ok) {
      document.getElementById("estadoVacio").style.display = "block";
      document.getElementById("mensajeError").textContent = data.error;
      return;
    }
    datosDias = data.dias;
    categoriasDisponibles = data.categorias || [];

    pintarSelectorFecha();
    poblarFiltroCategoria();
    pintarMetricas(data.metricas);
    aplicarFiltrosYRender();

    document.getElementById("contenidoPrincipal").style.display = "block";
    if (window.lucide) lucide.createIcons();
  } catch (e) {
    document.getElementById("estadoVacio").style.display = "block";
    document.getElementById("mensajeError").textContent = "Error al cargar: " + e.message;
  }
}

function pintarSelectorFecha() {
  const sel = document.getElementById("selectorFecha");
  sel.innerHTML = "";
  datosDias.forEach((d, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = new Date(d.fecha + "T00:00:00")
      .toLocaleDateString("es-EC", {weekday: "short", day: "numeric", month: "short", year: "numeric"});
    sel.appendChild(opt);
  });
  sel.onchange = () => { paginaActual = 1; aplicarFiltrosYRender(); };
}

function poblarFiltroCategoria() {
  const sel = document.getElementById("filtroCategoria");
  categoriasDisponibles.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c; opt.textContent = c;
    sel.appendChild(opt);
  });
}

function colorPrioridad(p) {
  return {"Alta": "#DC2626", "Media": "#D97706", "Baja": "#16A34A"}[p] || "#999";
}

function iconoCategoria(cat) {
  const mapa = {"Bebidas": "🥤", "Platos": "🍽️", "Acompañamientos": "🍟", "Postres": "🍰"};
  return mapa[cat] || "🍴";
}

function aplicarFiltrosYRender() {
  const idxDia = parseInt(document.getElementById("selectorFecha").value || 0);
  const dia = datosDias[idxDia];
  if (!dia) return;

  const texto = document.getElementById("buscar").value.toLowerCase();
  const prioridad = document.getElementById("filtroPrioridad").value;
  const categoria = document.getElementById("filtroCategoria").value;

  productosFiltrados = dia.productos.filter(p =>
    (!texto || p.producto.toLowerCase().includes(texto)) &&
    (!prioridad || p.prioridad === prioridad) &&
    (!categoria || p.categoria === categoria)
  );

  const conteo = {Alta: 0, Media: 0, Baja: 0};
  dia.productos.forEach(p => conteo[p.prioridad]++);
  const total = dia.productos.length || 1;
  document.getElementById("numAlta").textContent = conteo.Alta;
  document.getElementById("numMedia").textContent = conteo.Media;
  document.getElementById("numBaja").textContent = conteo.Baja;
  document.getElementById("numTotal").textContent = dia.productos.length;
  document.getElementById("pctAlta").textContent  = Math.round(100*conteo.Alta/total)  + "% del total";
  document.getElementById("pctMedia").textContent = Math.round(100*conteo.Media/total) + "% del total";
  document.getElementById("pctBaja").textContent  = Math.round(100*conteo.Baja/total)  + "% del total";

  pintarDonut(conteo);
  pintarTabla();
}

function pintarTabla() {
  const inicio = (paginaActual - 1) * POR_PAGINA;
  const pagina = productosFiltrados.slice(inicio, inicio + POR_PAGINA);

  const tbody = document.getElementById("tablaProductos");
  tbody.innerHTML = "";
  pagina.forEach(p => {
    const pct = Math.round((p.confianza || 0) * 100);
    const fila = document.createElement("tr");
    fila.innerHTML = `
      <td>${iconoCategoria(p.categoria)} ${p.producto}</td>
      <td>${p.categoria}</td>
      <td><span class="badge badge-${p.prioridad.toLowerCase()}">${p.prioridad}</span></td>
      <td>
        <div style="display:flex;align-items:center;">
          <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:${pct}%;background:${colorPrioridad(p.prioridad)};"></div></div>
          <span class="conf-pct">${pct}%</span>
        </div>
      </td>
      <td>${p.factores.map(f => `<span class="chip">${f}</span>`).join("")}</td>`;
    tbody.appendChild(fila);
  });

  const totalPaginas = Math.max(1, Math.ceil(productosFiltrados.length / POR_PAGINA));
  document.getElementById("resumenPaginacion").textContent =
    `Mostrando ${pagina.length ? inicio+1 : 0} a ${inicio + pagina.length} de ${productosFiltrados.length} productos`;

  const cont = document.getElementById("pagBotones");
  cont.innerHTML = "";
  for (let i = 1; i <= totalPaginas; i++) {
    const b = document.createElement("button");
    b.className = "pag-btn" + (i === paginaActual ? " active" : "");
    b.textContent = i;
    b.onclick = () => { paginaActual = i; pintarTabla(); };
    cont.appendChild(b);
  }
}

function pintarDonut(conteo) {
  const ctx = document.getElementById("donutPrioridad");
  if (chartDonut) chartDonut.destroy();
  chartDonut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Alta", "Media", "Baja"],
      datasets: [{ data: [conteo.Alta, conteo.Media, conteo.Baja],
                   backgroundColor: ["#DC2626", "#D97706", "#16A34A"], borderWidth: 0 }],
    },
    options: { plugins: { legend: { position: "bottom" } }, cutout: "65%" },
  });
}

function flechaDelta(valor) {
  if (valor === null || valor === undefined) return "";
  const clase = valor >= 0 ? "up" : "down";
  const signo = valor >= 0 ? "+" : "";
  const flecha = valor >= 0 ? "↑" : "↓";
  return `<span class="delta ${clase}">${flecha} ${signo}${valor.toFixed(3)}</span>`;
}

function pintarMetricas(m) {
  document.getElementById("mAcc").innerHTML = (m.accuracy != null ? (m.accuracy*100).toFixed(1) + "%" : "—")
    + (m.deltas ? flechaDelta(m.deltas.accuracy) : "");
  document.getElementById("mF1").innerHTML = (m.f1_macro != null ? m.f1_macro.toFixed(3) : "—")
    + (m.deltas ? flechaDelta(m.deltas.f1_macro) : "");

  window._matrizConfusion = m.matriz_confusion;
  window._matrizConfusionImg = m.matriz_confusion_img;
  window._clasesMatriz = m.clases;
}

function abrirMatriz() {
  document.getElementById("modalMatriz").classList.add("abierto");
  const img = window._matrizConfusionImg;
  document.getElementById("matrizImg").src = img || "";
}

async function reentrenar() {
  await fetch("/abastecimiento/reentrenar", {method: "POST"});
  cargar();
}

document.getElementById("buscar").addEventListener("input", () => { paginaActual = 1; aplicarFiltrosYRender(); });
document.getElementById("filtroPrioridad").addEventListener("change", () => { paginaActual = 1; aplicarFiltrosYRender(); });
document.getElementById("filtroCategoria").addEventListener("change", () => { paginaActual = 1; aplicarFiltrosYRender(); });

document.addEventListener("DOMContentLoaded", cargar);
</script>
{% endblock %}