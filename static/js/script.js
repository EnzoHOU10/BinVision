document.addEventListener('DOMContentLoaded', function() {
  const infoBlock = document.getElementById('info-block');
  const editBlock = document.getElementById('edit-block');
  const editBtn = document.getElementById('edit-btn');
  const cancelEditBtn = document.getElementById('cancel-edit-btn');
  if(editBlock) editBlock.style.display = 'none';
  if(editBtn) {
    editBtn.onclick = function() {
      if(infoBlock) infoBlock.style.display = 'none';
      if(editBlock) editBlock.style.display = 'block';
    };
  }
  if(cancelEditBtn) {
    cancelEditBtn.onclick = function() {
      if(editBlock) editBlock.style.display = 'none';
      if(infoBlock) infoBlock.style.display = 'block';
    };
  }
});
let files = [];
let currentIndex = 0;
let currentPage = 1;
const rowsPerPage = 8;
let rows = [];

document.getElementById("input").addEventListener("change", function (e) {
  files = Array.from(e.target.files);
  currentIndex = 0;
  showCurrentImage();
});

function showCurrentImage() {
  const preview = document.getElementById("preview");
  preview.innerHTML = "";
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
  fetch("/admin", {
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
  document.querySelector(".register").style.display = "none";
  document.querySelector(".details").style.display = "block";
}

function hideDetails() {
  document.querySelector(".details").style.display = "none";
  document.querySelector(".register").style.display = "block";
}

window.addEventListener("DOMContentLoaded", () => {
  const tbody = document.getElementById("tableaudimage");
  rows = Array.from(tbody.querySelectorAll("tr"));
  showPage(currentPage);
});

function showPage(page) {
  const totalPages = Math.ceil(rows.length / rowsPerPage);
  const start = (page - 1) * rowsPerPage;
  const end = start + rowsPerPage;

  rows.forEach((row, index) => {
    row.style.display = index >= start && index < end ? "" : "none";
  });

  document.getElementById(
    "pageInfo"
  ).textContent = `Page ${page}/${totalPages}`;
}

function nextPage() {
  if (currentPage * rowsPerPage < rows.length) {
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

function showSeuils() {
  document.querySelector(".seuils").style.display = "flex";
  document.querySelector('.seuils').scrollIntoView({behavior:'smooth'});
  document.querySelector(".content").style.display = "none";
}

function hideSeuils() {
  document.querySelector(".seuils").style.display = "none";
  document.querySelector(".content").style.display = "block";
}

function showImage(src) {
  const lightbox = document.getElementById("lightbox");
  const img = document.getElementById("lightbox-img");
  img.src = src;
  lightbox.style.display = "flex";
  document.body.style.overflow = "hidden";
}

function closeLightbox(event) {
  if (!event || event.target === event.currentTarget) {
    document.getElementById("lightbox").style.display = "none";
    document.body.style.overflow = "";
    document.getElementById("lightbox-img").src = "";
  }
}

document.addEventListener('DOMContentLoaded', function() {
  const preview = document.getElementById('preview');
  const loadingImg = document.getElementById('loading-img');
  const input = document.getElementById('input');
  function updateLoader() {
    if (!preview || !loadingImg) return;
    if (preview.children.length === 0) {
      loadingImg.style.display = 'block';
      preview.style.display = 'none';
    } else {
      loadingImg.style.display = 'none';
      preview.style.display = 'flex';
    }
  }
  if (input && preview && loadingImg) {
    loadingImg.style.display = 'block';
    preview.style.display = 'none';
    input.addEventListener('change', function() {
      setTimeout(updateLoader, 50);
    });
    preview.addEventListener('DOMSubtreeModified', updateLoader);
  }
});

