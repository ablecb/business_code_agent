(function () {
  try {
    var t = localStorage.getItem("bca_desktop_theme_v1") || "dark";
    if (t === "white") t = "light";
    if (t === "black") t = "dark";
    if (t !== "light" && t !== "dark") t = "dark";
    document.documentElement.setAttribute("data-theme", t);
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "dark");
  }
})();
