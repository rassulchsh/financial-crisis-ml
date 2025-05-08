# test_auth.py
import tweepy
import os

print("--- Attempting Authentication ---")

api_key = os.getenv("TWITTER_API_KEY")
api_secret = os.getenv("TWITTER_API_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

print(f"API Key: {api_key[:5]}..." if api_key else "Not Set")
print(f"API Secret: {api_secret[:5]}..." if api_secret else "Not Set")
print(f"Access Token: {access_token[:5]}..." if access_token else "Not Set")
print(
    f"Access Token Secret: {access_token_secret[:5]}..." if access_token_secret else "Not Set")


if not all([api_key, api_secret, access_token, access_token_secret]):
    print("\nError: One or more credentials are missing!")
    exit()

try:
    print("\nInitializing Tweepy Client...")
    client = tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )
    print("Client initialized.")
    print("\nAttempting client.get_me()...")
    response = client.get_me()
    print("\nAuthentication Successful!")
    print(f"User Data: {response.data}")

except tweepy.errors.TweepyException as e:
    print(f"\nAuthentication Failed!")
    print(f"Error: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

print("\n--- Test Complete ---")
