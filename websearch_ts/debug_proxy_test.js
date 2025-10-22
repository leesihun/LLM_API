// Proxy Environment Web Search Debugging Script
// Tests DuckDuckGo search with proxy detection and diagnostics

import fetch from 'node-fetch';
import * as cheerio from 'cheerio';
import { writeFileSync } from 'fs';

const COLORS = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m'
};

function log(color, prefix, message) {
  console.log(`${color}[${prefix}]${COLORS.reset} ${message}`);
}

async function testProxyEnvironment() {
  log(COLORS.blue, 'INFO', '=== Proxy Environment Detection ===');

  // Check proxy environment variables
  const proxyVars = {
    'HTTP_PROXY': process.env.HTTP_PROXY,
    'HTTPS_PROXY': process.env.HTTPS_PROXY,
    'http_proxy': process.env.http_proxy,
    'https_proxy': process.env.https_proxy,
    'NO_PROXY': process.env.NO_PROXY,
    'no_proxy': process.env.no_proxy
  };

  log(COLORS.yellow, 'ENV', 'Environment proxy settings:');
  for (const [key, value] of Object.entries(proxyVars)) {
    if (value) {
      log(COLORS.green, 'FOUND', `${key} = ${value}`);
    } else {
      log(COLORS.reset, 'NOT SET', `${key}`);
    }
  }

  console.log('');
}

async function testBasicConnectivity() {
  log(COLORS.blue, 'INFO', '=== Basic Connectivity Test ===');

  const testUrls = [
    'https://www.google.com',
    'https://html.duckduckgo.com',
    'https://www.bing.com'
  ];

  for (const url of testUrls) {
    try {
      const controller = new AbortController();
      setTimeout(() => controller.abort(), 5000);

      const start = Date.now();
      const response = await fetch(url, {
        signal: controller.signal,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      });
      const elapsed = Date.now() - start;

      const contentLength = response.headers.get('content-length') || 'unknown';

      if (response.ok) {
        log(COLORS.green, 'SUCCESS', `${url} - Status: ${response.status}, Size: ${contentLength} bytes, Time: ${elapsed}ms`);
      } else {
        log(COLORS.red, 'ERROR', `${url} - Status: ${response.status}`);
      }
    } catch (error) {
      log(COLORS.red, 'FAILED', `${url} - ${error.message}`);
    }
  }

  console.log('');
}

async function testDuckDuckGoWithDetails(query = 'python') {
  log(COLORS.blue, 'INFO', `=== DuckDuckGo Search Test: "${query}" ===`);

  const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;

  const testConfigs = [
    {
      name: 'With User-Agent',
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      }
    },
    {
      name: 'With Full Headers',
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
      }
    },
    {
      name: 'No Headers',
      headers: {}
    }
  ];

  for (const config of testConfigs) {
    try {
      log(COLORS.yellow, 'TEST', `Testing: ${config.name}`);

      const controller = new AbortController();
      setTimeout(() => controller.abort(), 10000);

      const start = Date.now();
      const response = await fetch(url, {
        signal: controller.signal,
        headers: config.headers
      });
      const elapsed = Date.now() - start;

      const htmlString = await response.text();
      const htmlSize = htmlString.length;

      // Save HTML for inspection
      const filename = `debug_ddg_${config.name.replace(/\s+/g, '_').toLowerCase()}.html`;
      writeFileSync(filename, htmlString);
      log(COLORS.blue, 'SAVED', `HTML saved to ${filename}`);

      // Parse results
      const $ = cheerio.load(htmlString);
      const results = $("div.results_links_deep");
      const resultCount = results.length;

      // Check for blocking/error patterns
      const blockingPatterns = {
        captcha: htmlString.includes('captcha') || htmlString.includes('CAPTCHA'),
        verify: htmlString.includes('verify you are human') || htmlString.includes('Verify'),
        blocked: htmlString.includes('access denied') || htmlString.includes('Access Denied'),
        cloudflare: htmlString.includes('cloudflare') && htmlString.includes('checking'),
        proxyError: htmlString.includes('proxy') && htmlString.includes('error'),
        connectionError: htmlString.includes('ERR_') || htmlString.includes('could not connect')
      };

      const hasBlocking = Object.values(blockingPatterns).some(v => v);

      // Response analysis
      log(COLORS.magenta, 'RESPONSE', `Status: ${response.status}, Size: ${htmlSize} bytes, Time: ${elapsed}ms`);
      log(COLORS.magenta, 'RESULTS', `Search results found: ${resultCount}`);

      if (hasBlocking) {
        log(COLORS.red, 'BLOCKING', 'Detected blocking patterns:');
        for (const [pattern, detected] of Object.entries(blockingPatterns)) {
          if (detected) {
            log(COLORS.red, 'BLOCKED', `  - ${pattern}`);
          }
        }
      }

      if (resultCount > 0) {
        log(COLORS.green, 'SUCCESS', `Found ${resultCount} results!`);

        // Show first 2 results
        results.slice(0, 2).each((i, result) => {
          const title = $(result).find("a.result__a").text();
          let link = $(result).find("a.result__snippet").attr("href");

          if (link) {
            link = link.replace("//duckduckgo.com/l/?uddg=", "").replace(/&rut=.*/, "");
            const decodedLink = decodeURIComponent(link);

            log(COLORS.green, `RESULT ${i + 1}`, `Title: ${title.substring(0, 60)}...`);
            log(COLORS.green, `RESULT ${i + 1}`, `URL: ${decodedLink.substring(0, 80)}...`);
          }
        });
      } else {
        log(COLORS.red, 'FAILED', 'No results found!');

        // Show HTML snippet for debugging
        const snippet = htmlString.substring(0, 500);
        log(COLORS.yellow, 'HTML', `First 500 chars: ${snippet}`);
      }

    } catch (error) {
      log(COLORS.red, 'ERROR', `${config.name} - ${error.message}`);

      if (error.code === 'ECONNREFUSED') {
        log(COLORS.red, 'HINT', 'Connection refused - possible firewall/proxy blocking');
      } else if (error.code === 'ETIMEDOUT' || error.name === 'AbortError') {
        log(COLORS.red, 'HINT', 'Timeout - slow network or proxy issues');
      } else if (error.code === 'ENOTFOUND') {
        log(COLORS.red, 'HINT', 'DNS resolution failed - check DNS/proxy settings');
      } else if (error.code === 'UNABLE_TO_VERIFY_LEAF_SIGNATURE') {
        log(COLORS.red, 'HINT', 'SSL certificate error - corporate proxy may intercept HTTPS');
      }
    }

    console.log('');
  }
}

async function testNetworkDiagnostics() {
  log(COLORS.blue, 'INFO', '=== Network Diagnostics ===');

  // Test DNS resolution
  try {
    const dns = await import('dns');
    const { promisify } = await import('util');
    const resolve4 = promisify(dns.resolve4);

    const hostname = 'html.duckduckgo.com';
    const addresses = await resolve4(hostname);
    log(COLORS.green, 'DNS', `${hostname} resolves to: ${addresses.join(', ')}`);
  } catch (error) {
    log(COLORS.red, 'DNS', `DNS resolution failed: ${error.message}`);
  }

  console.log('');
}

async function main() {
  console.log('\n' + '='.repeat(60));
  log(COLORS.magenta, 'START', 'Proxy Environment Web Search Debug');
  console.log('='.repeat(60) + '\n');

  // Run all tests
  await testProxyEnvironment();
  await testNetworkDiagnostics();
  await testBasicConnectivity();
  await testDuckDuckGoWithDetails('python programming');

  console.log('='.repeat(60));
  log(COLORS.magenta, 'DONE', 'Debug tests completed');
  log(COLORS.yellow, 'NOTE', 'Check saved HTML files for detailed inspection');
  console.log('='.repeat(60) + '\n');
}

main().catch(error => {
  log(COLORS.red, 'FATAL', error.message);
  console.error(error);
  process.exit(1);
});
