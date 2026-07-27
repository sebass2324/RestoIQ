// Toggle sidebar
const sidebar   = document.getElementById('sidebar');
const mainPanel = document.getElementById('main');
const toggleBtn = document.getElementById('toggleSidebar');

toggleBtn.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
  mainPanel.classList.toggle('expanded');
});

// Avatar inicial del nombre
const nameEl   = document.getElementById('userName');
const avatarEl = document.getElementById('userAvatar');
if (nameEl && nameEl.textContent.trim()) {
  avatarEl.textContent = nameEl.textContent.trim()[0].toUpperCase();
}

// Cuando haya datos, mostrar gráficas así:
// document.getElementById('chartVentasEmpty').style.display = 'none';
// document.getElementById('chartVentas').style.display = 'block';
// new Chart(document.getElementById('chartVentas'), { ... });