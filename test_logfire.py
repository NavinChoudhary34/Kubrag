
import logfire
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure Logfire
logfire.configure()

# Generate a test trace
with logfire.span("logfire_connection_test"):
    logfire.info("Testing Logfire connection")

print("Logfire test completed!")
