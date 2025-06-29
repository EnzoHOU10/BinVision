let files = [];
let currentIndex = 0;
let currentPage = 1;
const rowsPerPage = 5;
let rows = [];

document.getElementById('input').addEventListener("change", function (e) {
  files = Array.from(e.target.files);
  currentIndex = 0;
  showCurrentImage();
});

function showCurrentImage() {
  const preview = document.getElementById("preview");
  preview.innerHTML = '';
  if (currentIndex < files.length) {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(files[currentIndex]);
    preview.appendChild(img);
  } else {
    location.reload();
  }
}

function annotation(annotation) {
  if (currentIndex >= files.length) return;
  const formData = new FormData();
  formData.append("image", files[currentIndex]);
  formData.append("annotation", annotation);
  fetch("/", {
    method: "POST",
    body: formData,
  })
    .then((res) => {
      if (!res.ok) throw new Error("Erreur serveur");
      currentIndex++;
      showCurrentImage();
    })
    .catch((err) => alert("Erreur lors de l'envoi : " + err.message));
}

function showDetails() {
  document.querySelector('.register').style.display = 'none';
  document.querySelector('.details').style.display = 'block';
}

function hideDetails() {
  document.querySelector('.details').style.display = 'none';
  document.querySelector('.register').style.display = 'block';
}

function showImage(src) {
  const lightbox = document.getElementById('lightbox');
  const img = document.getElementById('lightbox-img');
  img.src = src;
  lightbox.style.display = 'flex';
}

function closeLightbox() {
  const lightbox = document.getElementById('lightbox');
  const img = document.getElementById('lightbox-img');
  img.src = '';
  lightbox.style.display = 'none';
}

window.addEventListener('DOMContentLoaded', () => {
  const tbody = document.getElementById('tableaudimage');
  rows = Array.from(tbody.querySelectorAll('tr'));
  showPage(currentPage);
});

function showPage(page) {
  const totalPages = Math.ceil(rows.length / rowsPerPage);
  const start = (page - 1) * rowsPerPage;
  const end = start + rowsPerPage;

  rows.forEach((row, index) => {
    row.style.display = index >= start && index < end ? '' : 'none';
  });

  document.getElementById('pageInfo').textContent = `Page ${page}/${totalPages}`;
}

function nextPage() {
  if ((currentPage * rowsPerPage) < rows.length) {
    currentPage++;
    showPage(currentPage);
  }
}

function prevPage() {
  if (currentPage > 1) {
    currentPage--;
    showPage(currentPage);
  }
}

