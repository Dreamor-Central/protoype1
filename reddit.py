import asyncpraw
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reddit API Credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")


async def fetch_reddit_posts(query, limit=5):
    """Searches Reddit globally for posts related to the query."""
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    try:
        subreddit = await reddit.subreddit("all")  # Ensure subreddit is awaited
        posts = []
        
        async for submission in subreddit.search(query, limit=limit, sort="relevance"):
            posts.append(
                {
                    "title": submission.title,
                    "url": f"https://reddit.com{submission.permalink}",
                    "score": submission.score,
                }
            )

        await reddit.close()
        return posts

    except Exception as e:
        print(f"❌ Error searching Reddit: {e}")
        return []

    finally:
        await reddit.close()


if __name__ == "__main__":
    asyncio.run(fetch_reddit_posts("AI trends 2025"))
