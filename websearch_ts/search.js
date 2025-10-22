// Page Assist Web Search - TypeScript to JavaScript (직접 복사)
import * as cheerio from 'cheerio';
import fetch from 'node-fetch';

// Google Search (Page Assist 원본)
async function localGoogleSearch(query, start = 0, domain = "google.com") {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 10000);

  try {
    const response = await fetch(
      `https://www.${domain}/search?hl=en&q=${encodeURIComponent(query)}&start=${start}`,
      {
        signal: controller.signal,
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
          "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
          "Accept-Language": "en-US,en;q=0.5",
          "Accept-Encoding": "gzip, deflate, br",
          "DNT": "1",
          "Connection": "keep-alive",
          "Upgrade-Insecure-Requests": "1",
          "Sec-Fetch-Dest": "document",
          "Sec-Fetch-Mode": "navigate",
          "Sec-Fetch-Site": "none",
          "Sec-Fetch-User": "?1"
        }
      }
    );

    const htmlString = await response.text();
    const $ = cheerio.load(htmlString);

    const searchResults = [];
    $("div.g").each((i, result) => {
      const title = $(result).find("h3").text();
      const link = $(result).find("a").attr("href");
      const content = $(result).find("span").map((i, span) => $(span).text()).get().join(" ");
      
      if (title && link) {
        searchResults.push({ title, link, content });
      }
    });

    return searchResults;
  } catch (error) {
    console.error('Google search error:', error.message);
    return [];
  }
}

async function webGoogleSearch(query, totalResults = 5, domain = "google.com") {
  let results = [];
  let currentPage = 0;
  const seenLinks = new Set();

  while (results.length < totalResults && currentPage < 10) {
    const start = currentPage * 10;
    const pageResults = await localGoogleSearch(query, start, domain);

    const uniquePageResults = pageResults.filter(result => {
      if (!result.link || seenLinks.has(result.link)) return false;
      seenLinks.add(result.link);
      return true;
    });

    results = [...results, ...uniquePageResults];

    if (pageResults.length === 0) break;
    
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
    currentPage++;
  }

  return results.slice(0, totalResults).map(r => ({
    title: r.title,
    url: r.link,
    snippet: r.content,
    source: "google"
  }));
}

// DuckDuckGo Search (Page Assist 원본)
async function webDuckDuckGoSearch(query, totalResults = 5) {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 10000);

  try {
    const response = await fetch(
      "https://html.duckduckgo.com/html/?q=" + encodeURIComponent(query),
      {
        signal: controller.signal,
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
      }
    );

    const htmlString = await response.text();
    const $ = cheerio.load(htmlString);

    const searchResults = [];
    $("div.results_links_deep").each((i, result) => {
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

    return searchResults.slice(0, totalResults);
  } catch (error) {
    console.error('DuckDuckGo search error:', error.message);
    return [];
  }
}

// Brave API Search (Page Assist 원본)
async function braveAPISearch(apiKey, query, totalResults = 5) {
  if (!apiKey || apiKey.trim() === "") {
    throw new Error("Brave API key not configured");
  }

  const searchURL = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${totalResults}`;
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 20000);

  try {
    const response = await fetch(searchURL, {
      signal: controller.signal,
      headers: {
        "X-Subscription-Token": apiKey,
        "Accept": "application/json",
      }
    });

    if (!response.ok) return [];

    const data = await response.json();
    return data?.web?.results.map(result => ({
      title: result.title,
      url: result.url,
      snippet: result.description,
      source: "brave_api"
    })) || [];
  } catch (error) {
    console.error('Brave API search error:', error.message);
    return [];
  }
}

// Tavily API Search (Page Assist 원본)
async function tavilyAPISearch(apiKey, query, totalResults = 5) {
  if (!apiKey || apiKey.trim() === "") {
    throw new Error("Tavily API key not configured");
  }

  const controller = new AbortController();
  setTimeout(() => controller.abort(), 20000);

  try {
    const response = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: apiKey,
        query,
        max_results: totalResults,
        include_answer: true
      }),
      signal: controller.signal
    });

    if (!response.ok) return [];

    const data = await response.json();
    return {
      answer: data.answer || '',
      results: data.results?.map(result => ({
        title: result.title || '',
        url: result.url || '',
        snippet: result.content || '',
        source: "tavily_api"
      })) || []
    };
  } catch (error) {
    console.error('Tavily API search error:', error.message);
    return [];
  }
}

// Exa API Search (Page Assist 원본)
async function exaAPISearch(apiKey, query, totalResults = 5) {
  if (!apiKey || apiKey.trim() === "") {
    throw new Error("Exa API key not configured");
  }

  const controller = new AbortController();
  setTimeout(() => controller.abort(), 20000);

  try {
    const response = await fetch("https://api.exa.ai/search", {
      signal: controller.signal,
      method: "POST",
      body: JSON.stringify({
        query,
        numResults: totalResults,
        text: true
      }),
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      }
    });

    if (!response.ok) return [];

    const data = await response.json();
    return data?.results?.map(result => ({
      title: result.title,
      url: result.url,
      snippet: result.text || result.title,
      source: "exa_api"
    })) || [];
  } catch (error) {
    console.error('Exa API search error:', error.message);
    return [];
  }
}

// Main search function
async function search(provider, query, config = {}) {
  const maxResults = config.max_results || 5;
  
  try {
    switch (provider) {
      case "google":
        return await webGoogleSearch(query, maxResults, config.google_domain || "google.com");
      
      case "duckduckgo":
        return await webDuckDuckGoSearch(query, maxResults);
      
      case "brave_api":
        return await braveAPISearch(config.brave_api_key, query, maxResults);
      
      case "tavily_api":
        return await tavilyAPISearch(config.tavily_api_key, query, maxResults);
      
      case "exa_api":
        return await exaAPISearch(config.exa_api_key, query, maxResults);
      
      default:
        // Fallback to Google
        return await webGoogleSearch(query, maxResults);
    }
  } catch (error) {
    console.error(`Search error with ${provider}:`, error.message);
    return [];
  }
}

// CLI Interface
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length < 2) {
    console.error(JSON.stringify({
      success: false,
      error: "Usage: node search.js <provider> <query> [config_json]"
    }));
    process.exit(1);
  }

  const provider = args[0];
  const query = args[1];
  const config = args[2] ? JSON.parse(args[2]) : {};

  try {
    const results = await search(provider, query, config);
    
    console.log(JSON.stringify({
      success: true,
      provider: provider,
      query: query,
      result_count: Array.isArray(results) ? results.length : (results.results?.length || 0),
      results: Array.isArray(results) ? results : results.results,
      answer: results.answer || null
    }));
  } catch (error) {
    console.error(JSON.stringify({
      success: false,
      error: error.message,
      provider: provider,
      query: query
    }));
    process.exit(1);
  }
}

main();

