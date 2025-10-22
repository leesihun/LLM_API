// 모든 provider를 테스트하고 상세한 디버그 정보 출력

import * as cheerio from 'cheerio';
import fetch from 'node-fetch';
import { writeFileSync } from 'fs';

async function debugProvider(name, url, selector, headerConfig = {}) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`Testing: ${name}`);
  console.log(`${'='.repeat(70)}`);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);

  try {
    console.log(`URL: ${url}`);
    console.log(`Sending request...`);

    const startTime = Date.now();
    const response = await fetch(url, {
      signal: controller.signal,
      headers: headerConfig
    });
    const requestTime = Date.now() - startTime;

    console.log(`✓ Response received in ${requestTime}ms`);
    console.log(`  Status: ${response.status}`);
    console.log(`  Status Text: ${response.statusText}`);

    const htmlString = await response.text();
    console.log(`  HTML Size: ${htmlString.length} bytes`);

    // Save HTML for inspection
    const filename = `debug_${name.replace(/\s+/g, '_')}.html`;
    writeFileSync(filename, htmlString);
    console.log(`  Saved to: ${filename}`);

    // Parse with Cheerio
    const $ = cheerio.load(htmlString);
    console.log(`  Title: "${$('title').text().trim()}"`);

    // Check for common blocking patterns
    const html = htmlString.toLowerCase();
    const blockers = {
      captcha: html.includes('captcha'),
      verify: html.includes('verify you are human') || html.includes('unusual traffic'),
      blocked: html.includes('access denied') || html.includes('forbidden'),
      cloudflare: html.includes('cloudflare') && html.includes('checking your browser'),
      recaptcha: html.includes('recaptcha')
    };

    let hasBlocker = false;
    for (const [key, value] of Object.entries(blockers)) {
      if (value) {
        console.log(`  ⚠️  Detected: ${key.toUpperCase()}`);
        hasBlocker = true;
      }
    }

    if (!hasBlocker) {
      console.log(`  ✓ No blocking detected`);
    }

    // Try to find results
    const results = $(selector);
    console.log(`  Results found (${selector}): ${results.length}`);

    if (results.length > 0) {
      console.log(`  ✓ SUCCESS - Found ${results.length} results`);

      // Show first result details
      const first = results.first();
      console.log(`\n  First result sample:`);
      console.log(`    HTML: ${first.html()?.substring(0, 200)}...`);
    } else {
      console.log(`  ✗ FAILED - No results found`);

      // Try to find what's actually in the page
      console.log(`\n  Trying to diagnose...`);
      console.log(`    <div> count: ${$('div').length}`);
      console.log(`    <a> count: ${$('a').length}`);
      console.log(`    <h1> text: "${$('h1').first().text()}"`);
      console.log(`    <h2> text: "${$('h2').first().text()}"`);
    }

  } catch (error) {
    console.log(`  ✗ ERROR: ${error.message}`);

    if (error.name === 'AbortError') {
      console.log(`  Reason: Request timed out after 15 seconds`);
      console.log(`  Possible causes:`);
      console.log(`    - Network is too slow`);
      console.log(`    - Firewall blocking`);
      console.log(`    - DNS issues`);
    }
  } finally {
    clearTimeout(timeout);
  }
}

async function runAllTests() {
  console.log('\n' + '='.repeat(70));
  console.log('Web Search Provider Debug Tool');
  console.log('='.repeat(70));

  // Test 1: Google
  await debugProvider(
    'Google',
    'https://www.google.com/search?hl=en&q=python+tutorial',
    'div.g',
    {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
      "Accept-Language": "en-US,en;q=0.5"
    }
  );

  // Test 2: DuckDuckGo
  await debugProvider(
    'DuckDuckGo',
    'https://html.duckduckgo.com/html/?q=python+tutorial',
    'div.results_links_deep',
    {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
  );

  // Test 3: Google (no headers)
  await debugProvider(
    'Google (No Headers)',
    'https://www.google.com/search?hl=en&q=python+tutorial',
    'div.g',
    {}
  );

  // Test 4: DuckDuckGo (no headers)
  await debugProvider(
    'DuckDuckGo (No Headers)',
    'https://html.duckduckgo.com/html/?q=python+tutorial',
    'div.results_links_deep',
    {}
  );

  console.log('\n' + '='.repeat(70));
  console.log('Testing Complete');
  console.log('='.repeat(70));
  console.log('\nCheck the debug_*.html files to see what was actually returned.');
  console.log('');
}

runAllTests().catch(console.error);
