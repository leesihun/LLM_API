// Simple Web Search Test - Minimal script for quick testing
// Usage: node test_search_simple.js <query>

import * as cheerio from 'cheerio';
import fetch from 'node-fetch';

async function quickDuckDuckGoSearch(query) {
  console.log(`\n[SEARCH] Query: "${query}"`);
  console.log('[SEARCH] Provider: DuckDuckGo');
  console.log('[SEARCH] URL: https://html.duckduckgo.com/html/');
  console.log('');

  try {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 10000);

    const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;

    console.log('[FETCH] Sending request...');
    const start = Date.now();

    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      }
    });

    const elapsed = Date.now() - start;
    console.log(`[FETCH] Response received in ${elapsed}ms`);
    console.log(`[FETCH] Status: ${response.status}`);

    const htmlString = await response.text();
    console.log(`[FETCH] HTML size: ${htmlString.length} bytes`);
    console.log('');

    // Parse results
    const $ = cheerio.load(htmlString);
    const results = $("div.results_links_deep");

    console.log(`[PARSE] Found ${results.length} search results`);
    console.log('');

    if (results.length === 0) {
      console.log('[ERROR] No results found!');
      console.log('[DEBUG] HTML snippet:');
      console.log(htmlString.substring(0, 500));
      console.log('...\n');
      return [];
    }

    // Extract results
    const searchResults = [];
    results.each((i, result) => {
      const title = $(result).find("a.result__a").text();
      let link = $(result).find("a.result__snippet").attr("href");
      const content = $(result).find("a.result__snippet").text();

      if (link) {
        link = link.replace("//duckduckgo.com/l/?uddg=", "").replace(/&rut=.*/, "");
        const decodedLink = decodeURIComponent(link);

        if (title && decodedLink) {
          searchResults.push({
            title,
            url: decodedLink,
            snippet: content,
            source: "duckduckgo"
          });
        }
      }
    });

    console.log('[SUCCESS] Extracted results:');
    searchResults.forEach((result, i) => {
      console.log(`\n[${i + 1}] ${result.title}`);
      console.log(`    URL: ${result.url}`);
      console.log(`    Snippet: ${result.snippet.substring(0, 100)}...`);
    });

    return searchResults;

  } catch (error) {
    console.log(`\n[ERROR] Search failed: ${error.message}`);
    console.log(`[ERROR] Error code: ${error.code || 'N/A'}`);

    // Provide troubleshooting hints
    if (error.code === 'ECONNREFUSED') {
      console.log('[HINT] Connection refused - firewall or proxy may be blocking');
    } else if (error.code === 'ETIMEDOUT' || error.name === 'AbortError') {
      console.log('[HINT] Request timed out - check network/proxy settings');
    } else if (error.code === 'ENOTFOUND') {
      console.log('[HINT] DNS lookup failed - check DNS/proxy configuration');
    } else if (error.message.includes('certificate')) {
      console.log('[HINT] SSL certificate error - corporate proxy may intercept HTTPS');
      console.log('[HINT] Try setting NODE_TLS_REJECT_UNAUTHORIZED=0 (for testing only!)');
    }

    return [];
  }
}

// Main execution
const query = process.argv[2] || 'python programming';
console.log('='.repeat(60));
console.log('Simple Web Search Test');
console.log('='.repeat(60));

quickDuckDuckGoSearch(query)
  .then(results => {
    console.log('\n' + '='.repeat(60));
    console.log(`FINAL RESULT: ${results.length} results found`);
    console.log('='.repeat(60) + '\n');
    process.exit(results.length > 0 ? 0 : 1);
  })
  .catch(error => {
    console.error('\n[FATAL]', error);
    process.exit(1);
  });
