import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late final WebViewController _controller;
  final TextEditingController _urlController = TextEditingController();
  bool _isLoading = false;
  double _progress = 0;
  bool _isWebViewVisible = false;
  String _appName = "Home";

  @override
  void initState() {
    super.initState();
    _controller =
        WebViewController()
          ..setJavaScriptMode(JavaScriptMode.unrestricted)
          ..setBackgroundColor(const Color(0x00000000))
          ..setNavigationDelegate(
            NavigationDelegate(
              onProgress: (int progress) {
                setState(() {
                  _progress = progress / 100;
                });
              },
              onPageStarted: (String url) {
                setState(() {
                  _isLoading = true;
                });
              },
              onPageFinished: (String url) {
                setState(() {
                  _isLoading = false;
                });
              },
              onWebResourceError: (WebResourceError error) {},
              onNavigationRequest: (NavigationRequest request) {
                return NavigationDecision.navigate;
              },
            ),
          );

    _loadCustomSettings();
  }

  void _loadCustomSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _appName = prefs.getString('custom_app_name') ?? "Home";
    });
  }

  void _loadUrl(String url) async {
    if (url.isEmpty) return;

    String formattedUrl = url;
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      formattedUrl = 'https://$url';
    }

    setState(() {
      _isWebViewVisible = true;
    });

    _controller.loadRequest(Uri.parse(formattedUrl));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final primaryColor = theme.colorScheme.primary;

    SystemChrome.setSystemUIOverlayStyle(
      SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: isDark ? Brightness.light : Brightness.dark,
      ),
    );

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (bool didPop, dynamic result) async {
        if (didPop) return;
        if (_isWebViewVisible) {
          if (await _controller.canGoBack()) {
            await _controller.goBack();
          } else {
            setState(() {
              _isWebViewVisible = false;
            });
          }
        } else {
          SystemNavigator.pop();
        }
      },
      child: Scaffold(
        appBar: AppBar(
          elevation: 0,
          title: Text(
            _appName,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: theme.colorScheme.onSurface,
            ),
          ),
          actions: [
            if (_isWebViewVisible)
              IconButton(
                icon: Icon(Icons.refresh, color: primaryColor),
                onPressed: () => _controller.reload(),
              ),
            IconButton(
              icon: Icon(Icons.tune_rounded, color: primaryColor),
              onPressed: () async {
                final result = await Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const SettingsScreen(),
                  ),
                );
                if (result == true) _loadCustomSettings();
              },
            ),
            IconButton(
              icon: const Icon(Icons.logout, color: Colors.redAccent),
              onPressed: () => FirebaseAuth.instance.signOut(),
            ),
          ],
        ),
        body: SafeArea(
          child: Stack(
            children: [
              if (!_isWebViewVisible)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.phonelink_setup_rounded,
                        size: 80,
                        color: primaryColor,
                      ),
                      const SizedBox(height: 24),
                      Text(
                        'Convert Website to App',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: theme.colorScheme.onSurface,
                        ),
                      ),
                      const SizedBox(height: 32),
                      Container(
                        decoration: BoxDecoration(
                          color:
                              isDark
                                  ? Colors.grey[900]
                                  : const Color(0xFFF1F5F9),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color:
                                isDark
                                    ? Colors.grey[800]!
                                    : const Color(0xFFE2E8F0),
                          ),
                        ),
                        child: TextField(
                          controller: _urlController,
                          style: TextStyle(color: theme.colorScheme.onSurface),
                          decoration: InputDecoration(
                            hintText: 'e.g. google.com',
                            hintStyle: TextStyle(
                              color: theme.colorScheme.onSurface.withOpacity(
                                0.5,
                              ),
                            ),
                            prefixIcon: Icon(
                              Icons.language,
                              color: primaryColor,
                            ),
                            border: InputBorder.none,
                            contentPadding: const EdgeInsets.symmetric(
                              vertical: 18,
                            ),
                            suffixIcon: IconButton(
                              icon: Icon(
                                Icons.rocket_launch_rounded,
                                color: primaryColor,
                              ),
                              onPressed: () => _loadUrl(_urlController.text),
                            ),
                          ),
                          onSubmitted: _loadUrl,
                        ),
                      ),
                    ],
                  ),
                )
              else
                // Using WebViewWidget directly without the SingleChildScrollView wrapper
                // fix the scroll issues as WebView handles its own scrolling.
                WebViewWidget(controller: _controller),

              if (_isWebViewVisible) ...[
                if (_isLoading)
                  LinearProgressIndicator(
                    value: _progress,
                    backgroundColor: Colors.transparent,
                    color: primaryColor,
                    minHeight: 3,
                  ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}