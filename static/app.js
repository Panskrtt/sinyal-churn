// Tombol yang memicu proses lama menampilkan status dan tidak bisa diklik dua kali.
document.querySelectorAll("form[data-busy]").forEach(function (f) {
  f.addEventListener("submit", function () {
    var b = f.querySelector("button[type=submit]");
    if (b) { b.disabled = true; b.textContent = f.dataset.busy; }
  });
});
