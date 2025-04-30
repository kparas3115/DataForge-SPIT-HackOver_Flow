// services/news.service.js
const Parser = require("rss-parser");
const parser = new Parser();

/**
 * List of RSS feeds for research news
 */
const RSS_FEEDS = [
    {
        name: "Nature",
        url: "https://www.nature.com/nature.rss",
        category: "science",
    },
    {
        name: "MIT Technology Review",
        url: "https://www.technologyreview.com/feed/",
        category: "technology",
    },
    {
        name: "arXiv CS",
        url: "http://export.arxiv.org/rss/cs",
        category: "computer science",
    },
    {
        name: "PNAS",
        url: "https://www.pnas.org/action/showFeed?type=etoc&feed=rss&jc=pnas",
        category: "science",
    },
];

/**
 * Fetch news from a single RSS feed
 * @param {Object} feed - Feed object with name and URL
 * @returns {Array} - Array of news items
 */
const fetchFeedNews = async (feed) => {
    try {
        const feedData = await parser.parseURL(feed.url);

        // Process and normalize feed items
        return feedData.items
            .map((item) => ({
                title: item.title,
                link: item.link,
                pubDate: item.pubDate || item.isoDate,
                content: item.contentSnippet || item.summary || "",
                source: feed.name,
                category: feed.category,
                guid: item.guid || item.id || item.link,
            }))
            .slice(0, 10); // Limit to 10 items per feed
    } catch (error) {
        console.error(`Error fetching ${feed.name} feed:`, error);
        return []; // Return empty array on error
    }
};

/**
 * Fetch news from all configured RSS feeds
 * @param {Object} options - Options for filtering news
 * @returns {Object} - Result with news items or error
 */
const getResearchNews = async (options = {}) => {
    try {
        const { category, limit = 50 } = options;

        // Filter feeds by category if specified
        const feedsToFetch = category
            ? RSS_FEEDS.filter(
                  (feed) =>
                      feed.category.toLowerCase() === category.toLowerCase()
              )
            : RSS_FEEDS;

        if (feedsToFetch.length === 0) {
            return {
                success: false,
                status: 404,
                message: "No feeds found for the specified category",
            };
        }

        // Fetch news from all selected feeds in parallel
        const feedPromises = feedsToFetch.map((feed) => fetchFeedNews(feed));
        const feedResults = await Promise.all(feedPromises);

        // Combine all feed results
        let allNews = feedResults.flat();

        // Sort by publication date (newest first)
        allNews.sort((a, b) => new Date(b.pubDate) - new Date(a.pubDate));

        // Apply limit
        allNews = allNews.slice(0, limit);

        return {
            success: true,
            status: 200,
            news: allNews,
            meta: {
                total: allNews.length,
                sources: feedsToFetch.map((feed) => feed.name),
            },
        };
    } catch (error) {
        console.error("Get Research News Error:", error);
        return {
            success: false,
            status: 500,
            message: "Error fetching research news",
        };
    }
};

module.exports = {
    getResearchNews,
};
