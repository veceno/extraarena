const SPACE_HOST = "app.extraarena.space";
const RU_HOST = "app.laveqox.ru";
const RU_REDIRECT_STATUS = 302;
const REDIRECT_BYPASS_PATHS = ["/ready", "/api", "/socket.io"];

function bypassRegionalRedirect(pathname) {
  return REDIRECT_BYPASS_PATHS.some((prefix) => (
    pathname === prefix || pathname.startsWith(`${prefix}/`)
  ));
}

export default {
  fetch(request) {
    const url = new URL(request.url);
    const country = request.cf?.country || request.headers.get("CF-IPCountry") || "";

    if (
      url.hostname === SPACE_HOST &&
      country === "RU" &&
      !bypassRegionalRedirect(url.pathname)
    ) {
      url.hostname = RU_HOST;
      url.protocol = "https:";
      return Response.redirect(url.toString(), RU_REDIRECT_STATUS);
    }

    return fetch(request);
  },
};
