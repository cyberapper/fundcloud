/* Fundcloud docs — small behaviour layer.
   Intentionally minimal; the design system does the work in CSS. */

(() => {
  "use strict";

  /* 1. Fade-in for hero + cards once they enter the viewport */
  const io = "IntersectionObserver" in window
    ? new IntersectionObserver((entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.setAttribute("data-fc-visible", "true");
            io.unobserve(e.target);
          }
        }
      }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" })
    : null;

  const mark = () => {
    const targets = document.querySelectorAll(
      ".fc-hero, .fc-track, .fc-card, .fc-cta"
    );
    targets.forEach((el) => {
      el.setAttribute("data-fc-fade", "true");
      if (io) io.observe(el);
      else el.setAttribute("data-fc-visible", "true");
    });
  };

  /* 2. Append a subtle "#" anchor link to h2/h3 on hover — cheap polish */
  const addAnchors = () => {
    const article = document.querySelector("article.md-content__inner");
    if (!article) return;
    article.querySelectorAll("h2[id], h3[id]").forEach((h) => {
      if (h.querySelector(".fc-anchor")) return;
      const a = document.createElement("a");
      a.href = "#" + h.id;
      a.className = "fc-anchor";
      a.setAttribute("aria-label", "Permalink");
      a.textContent = "#";
      h.appendChild(a);
    });
  };

  const run = () => { mark(); addAnchors(); };

  /* Material loads pages via instant navigation — rehook. */
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(run);
  } else {
    document.addEventListener("DOMContentLoaded", run);
  }
})();
