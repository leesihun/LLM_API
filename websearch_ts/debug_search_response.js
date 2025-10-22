// 실제로 Google/DuckDuckGo가 뭘 보내는지 확인하는 스크립트

import fetch from 'node-fetch';
import * as cheerio from 'cheerio';
import { writeFileSync } from 'fs';

async function debugDuckDuckGoResponse() {
  console.log('='.repeat(70));
  console.log('DuckDuckGo Response Debug');
  console.log('='.repeat(70));

  try {
    const response = await fetch(
      "https://html.duckduckgo.com/html/?q=python",
      {
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
      }
    );

    const htmlString = await response.text();

    console.log('\nResponse Status:', response.status);
    console.log('Response Headers:', response.headers.raw());
    console.log('\nHTML Length:', htmlString.length);

    // HTML 저장
    writeFileSync('duckduckgo_response.html', htmlString);
    console.log('\n✓ Saved to: duckduckgo_response.html');

    // Cheerio로 파싱
    const $ = cheerio.load(htmlString);

    console.log('\n--- HTML Analysis ---');
    console.log('Title:', $('title').text());
    console.log('Search results found:', $("div.results_links_deep").length);
    console.log('Links found:', $("a.result__a").length);

    // CAPTCHA 확인
    const hasCaptcha = htmlString.toLowerCase().includes('captcha') ||
                      htmlString.toLowerCase().includes('verify') ||
                      htmlString.toLowerCase().includes('unusual traffic');

    console.log('Contains CAPTCHA?', hasCaptcha);

    if (hasCaptcha) {
      console.log('\n⚠️  CAPTCHA DETECTED!');
      console.log('This is why we get no results.');
      console.log('The HTML contains verification page, not search results.');
    }

    // 실제 검색 결과 추출 시도
    console.log('\n--- Attempting to extract results ---');
    const searchResults = [];
    $("div.results_links_deep").each((i, result) => {
      const title = $(result).find("a.result__a").text();
      const link = $(result).find("a.result__snippet").attr("href");

      if (title && link) {
        searchResults.push({ title, link });
      }
    });

    console.log('Results extracted:', searchResults.length);
    if (searchResults.length > 0) {
      console.log('\n✓ Sample result:');
      console.log('  Title:', searchResults[0].title);
      console.log('  Link:', searchResults[0].link);
    } else {
      console.log('\n✗ No results extracted');
      console.log('  Reason: HTML structure not matching expected format');
    }

  } catch (error) {
    console.error('Error:', error.message);
  }

  console.log('\n' + '='.repeat(70));
}

debugDuckDuckGoResponse();
