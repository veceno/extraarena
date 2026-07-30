import SwiftUI
import WebKit

@main
struct ExtraArenaHarnessApp: App {
    var body: some Scene {
        WindowGroup {
            TelegramFullsizeContainer()
        }
    }
}

private struct TelegramFullsizeContainer: View {
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 9) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 17, weight: .semibold))
                Text("ExtraArena")
                    .font(.system(size: 17, weight: .semibold))
                Spacer()
                Text("Fullsize")
                    .font(.system(size: 11, weight: .semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.12), in: Capsule())
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .frame(height: 48)
            .background(Color(red: 0.075, green: 0.09, blue: 0.11))

            TelegramWebView(url: LaunchConfiguration.url)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Color.black)
        .ignoresSafeArea(edges: .bottom)
    }
}

private enum LaunchConfiguration {
    static let userID = ProcessInfo.processInfo.environment["EXTRAARENA_USER_ID"] ?? "9000000001"

    static let url: URL = {
        if let configured = ProcessInfo.processInfo.environment["EXTRAARENA_URL"],
           let url = URL(string: configured) {
            return url
        }

        var components = URLComponents(string: "http://127.0.0.1:8081/")!
        components.queryItems = [
            URLQueryItem(name: "user_id", value: userID),
            URLQueryItem(name: "ios_simulator", value: "1"),
        ]
        return components.url!
    }()
}

private struct TelegramWebView: UIViewRepresentable {
    let url: URL

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        configuration.userContentController.addUserScript(
            WKUserScript(
                source: VendorScripts.source + "\n" + TelegramMock.script(userID: LaunchConfiguration.userID),
                injectionTime: .atDocumentStart,
                forMainFrameOnly: true
            )
        )
        configuration.userContentController.add(context.coordinator, name: "extraArenaQA")

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        webView.scrollView.bounces = false
        webView.isOpaque = false
        webView.backgroundColor = .black
        webView.scrollView.backgroundColor = .black
        if #available(iOS 16.4, *) {
            webView.isInspectable = true
        }
        webView.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
        context.coordinator.webView = webView
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        weak var webView: WKWebView?

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            print("[ExtraArenaHarness] loaded \(webView.url?.absoluteString ?? "unknown URL")")
        }

        func webView(
            _ webView: WKWebView,
            didFail navigation: WKNavigation!,
            withError error: Error
        ) {
            print("[ExtraArenaHarness] navigation failed: \(error.localizedDescription)")
        }

        func webView(
            _ webView: WKWebView,
            didFailProvisionalNavigation navigation: WKNavigation!,
            withError error: Error
        ) {
            print("[ExtraArenaHarness] provisional navigation failed: \(error.localizedDescription)")
        }

        func userContentController(
            _ userContentController: WKUserContentController,
            didReceive message: WKScriptMessage
        ) {
            print("[ExtraArenaHarness][QA] \(message.body)")
        }
    }
}

private enum TelegramMock {
    static func script(userID: String) -> String {
        let numericID = Int64(userID) ?? 9_000_000_001
        return """
        (() => {
          const listeners = Object.create(null);
          const emit = (name, payload) => {
            (listeners[name] || []).slice().forEach((callback) => {
              try { callback(payload); } catch (error) { console.error(error); }
            });
          };
          const button = {
            isVisible: false,
            isActive: true,
            text: '',
            color: '#2481cc',
            textColor: '#ffffff',
            show() { this.isVisible = true; return this; },
            hide() { this.isVisible = false; return this; },
            enable() { this.isActive = true; return this; },
            disable() { this.isActive = false; return this; },
            setText(value) { this.text = String(value || ''); return this; },
            setParams(params) { Object.assign(this, params || {}); return this; },
            onClick(callback) { this._callback = callback; return this; },
            offClick(callback) { if (this._callback === callback) this._callback = null; return this; },
            showProgress() { return this; },
            hideProgress() { return this; },
          };
          const backButton = {
            isVisible: false,
            show() { this.isVisible = true; return this; },
            hide() { this.isVisible = false; return this; },
            onClick(callback) { this._callback = callback; return this; },
            offClick(callback) { if (this._callback === callback) this._callback = null; return this; },
          };
          const webApp = {
            initData: '',
            initDataUnsafe: {
              user: {
                id: \(numericID),
                first_name: 'iOS',
                last_name: 'Simulator',
                username: 'extraarena_ios_qa',
                language_code: 'ru',
              },
            },
            version: '9.2',
            platform: 'ios',
            colorScheme: 'dark',
            themeParams: {
              bg_color: '#07040f',
              text_color: '#ffffff',
              hint_color: '#9ca3af',
              link_color: '#62a8ea',
              button_color: '#2481cc',
              button_text_color: '#ffffff',
              secondary_bg_color: '#131820',
            },
            isExpanded: true,
            get isFullscreen() { return false; },
            get viewportHeight() { return Math.max(1, window.innerHeight + 120); },
            get viewportStableHeight() { return Math.max(1, window.innerHeight + 120); },
            safeAreaInset: { top: 0, right: 0, bottom: 34, left: 0 },
            contentSafeAreaInset: { top: 0, right: 0, bottom: 34, left: 0 },
            headerColor: '#131820',
            backgroundColor: '#07040f',
            bottomBarColor: '#07040f',
            isClosingConfirmationEnabled: false,
            isVerticalSwipesEnabled: true,
            BackButton: backButton,
            MainButton: Object.create(button),
            SecondaryButton: Object.create(button),
            SettingsButton: Object.create(backButton),
            HapticFeedback: {
              impactOccurred() {},
              notificationOccurred() {},
              selectionChanged() {},
            },
            CloudStorage: {
              setItem(key, value, callback) { callback?.(null, true); },
              getItem(key, callback) { callback?.(null, ''); },
              getItems(keys, callback) { callback?.(null, {}); },
              removeItem(key, callback) { callback?.(null, true); },
              removeItems(keys, callback) { callback?.(null, true); },
              getKeys(callback) { callback?.(null, []); },
            },
            ready() {},
            expand() { emit('viewportChanged', { isStateStable: true }); },
            close() {},
            enableClosingConfirmation() { this.isClosingConfirmationEnabled = true; },
            disableClosingConfirmation() { this.isClosingConfirmationEnabled = false; },
            enableVerticalSwipes() { this.isVerticalSwipesEnabled = true; },
            disableVerticalSwipes() { this.isVerticalSwipesEnabled = false; },
            setHeaderColor(color) { this.headerColor = color; },
            setBackgroundColor(color) { this.backgroundColor = color; },
            setBottomBarColor(color) { this.bottomBarColor = color; },
            isVersionAtLeast() { return true; },
            onEvent(name, callback) {
              if (typeof callback !== 'function') return this;
              (listeners[name] ||= []).push(callback);
              return this;
            },
            offEvent(name, callback) {
              if (!listeners[name]) return this;
              listeners[name] = listeners[name].filter((entry) => entry !== callback);
              return this;
            },
            requestFullscreen() {
              window.webkit?.messageHandlers?.extraArenaQA?.postMessage({
                event: 'requestFullscreen',
                result: 'rejected-to-reproduce-ios-fullsize',
              });
              setTimeout(() => emit('fullscreenFailed', { error: 'UNSUPPORTED' }), 0);
            },
            exitFullscreen() {},
            openLink() {},
            openTelegramLink() {},
            openInvoice(url, callback) { callback?.('cancelled'); },
            showPopup(params, callback) { callback?.(params?.buttons?.[0]?.id || null); },
            showAlert(message, callback) { callback?.(); },
            showConfirm(message, callback) { callback?.(false); },
          };

          const telegram = window.Telegram && typeof window.Telegram === 'object'
            ? window.Telegram
            : {};
          Object.defineProperty(telegram, 'WebApp', {
            configurable: true,
            enumerable: true,
            get: () => webApp,
            set: () => {},
          });
          Object.defineProperty(window, 'Telegram', {
            configurable: true,
            enumerable: true,
            get: () => telegram,
            set: (value) => {
              if (!value || typeof value !== 'object') return;
              Object.keys(value).forEach((key) => {
                if (key !== 'WebApp') telegram[key] = value[key];
              });
            },
          });

          const reportError = (kind, detail) => {
            window.webkit?.messageHandlers?.extraArenaQA?.postMessage({
              event: 'javascript-error',
              kind,
              detail: String(detail || 'unknown error').slice(0, 800),
            });
          };
          window.addEventListener('error', (event) => {
            reportError(
              'error',
              `${event.message || 'unknown error'} at ${event.filename || 'unknown'}:${event.lineno || 0}:${event.colno || 0}\n${event.error?.stack || ''}`
            );
          });
          window.addEventListener('unhandledrejection', (event) => {
            reportError('unhandledrejection', event.reason?.stack || event.reason);
          });

          const report = (reason) => {
            const root = document.documentElement;
            const appRoot = document.getElementById('root');
            const visibleButtons = Array.from(document.querySelectorAll('button'))
              .filter((element) => {
                const rect = element.getBoundingClientRect();
                const style = getComputedStyle(element);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden';
              })
              .map((element) => {
                const rect = element.getBoundingClientRect();
                return {
                  text: (element.innerText || element.getAttribute('aria-label') || '').trim().slice(0, 60),
                  top: Math.round(rect.top),
                  bottom: Math.round(rect.bottom),
                  height: Math.round(rect.height),
                };
              })
              .slice(-12);
            window.webkit?.messageHandlers?.extraArenaQA?.postMessage({
              event: 'layout',
              reason,
              path: location.pathname + location.search,
              readyState: document.readyState,
              width: window.innerWidth,
              innerHeight: window.innerHeight,
              visualHeight: Math.round(window.visualViewport?.height || 0),
              telegramViewportHeight: webApp.viewportStableHeight,
              cssViewportHeight: getComputedStyle(root).getPropertyValue('--ea-viewport-height').trim(),
              classes: root.className,
              appRootChildren: appRoot?.children?.length || 0,
              bodyText: (document.body?.innerText || '').trim().slice(0, 400),
              visibleButtons,
            });
          };
          window.__extraArenaHarnessReport = report;
          window.addEventListener('resize', () => emit('viewportChanged', { isStateStable: true }));
          window.addEventListener('load', () => {
            setTimeout(() => report('load+1s'), 1000);
            setTimeout(() => report('load+4s'), 4000);
          });
        })();
        """
    }
}

private enum VendorScripts {
    static let source: String = [
        "react.production.min",
        "react-dom.production.min",
        "purify.min",
        "socket.io.min",
    ].compactMap { name in
        guard let url = Bundle.main.url(forResource: name, withExtension: "js") else {
            print("[ExtraArenaHarness] missing bundled script: \(name).js")
            return nil
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }.joined(separator: "\n")
}
