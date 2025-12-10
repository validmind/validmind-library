import pandas as pd
import uuid

# Banking-specific test dataset for retail and commercial banking
# tools: credit_risk_analyzer, customer_account_manager, fraud_detection_system
banking_test_dataset = pd.DataFrame([
    {
        "input": "Analyze credit risk for a $50,000 personal loan application with $75,000 annual income, $1,200 monthly debt, and 720 credit score",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["LOW RISK", "APPROVE", "risk score", "720", "probability of default", "2.5%"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    },
    {
        "input": "Evaluate credit risk for a business loan of $250,000 with monthly revenue of $85,000 and existing debt of $45,000 and credit score of 650",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["MEDIUM RISK", "HIGH RISK", "business loan", "debt service coverage ratio", "1.8", "annual revenue", "$1,020,000", "risk score", "650"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    },
    {
        "input": "Check account balance for checking account 12345",
        "expected_tools": ["customer_account_manager"],
        # possible_outputs values relevant to account management
        # Matches what _handle_check_balance would return for customer 12345 ("John Smith"), whose checking_balance is 2547.89 in the mock DB.
        "possible_outputs": ["$2,547.89", "John Smith", "$2547.89"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Analyze fraud risk for a $15,000 wire transfer from customer 67890 to Nigeria",
        "expected_tools": ["fraud_detection_system"],
        "possible_outputs": ["REQUIRE VERIFICATION", "fraud score", "65", "geographic risk", "block transaction", "MEDIUM RISK"],
        "session_id": str(uuid.uuid4()),
        "category": "fraud_detection"
    },
    {
        "input": "Recommend banking products for customer 11111 with $150,000 in savings and 720 credit score",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["High-Yield Savings Account (2.5% APY)", "Personal Line of Credit up to $25,000"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Investigate suspicious transactions totaling $75,000 across multiple accounts in the last week",
        "expected_tools": ["fraud_detection_system"],
        "possible_outputs": ["Require additional verification", "Implement 24-hour delay for verification"],
        "session_id": str(uuid.uuid4()),
        "category": "fraud_detection"
    },
    {
        "input": "Assess credit risk for a $1,000,000 commercial real estate loan with $500,000 annual business income",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["HIGH RISK", "VERY HIGH RISK","loan-to-value", "66.7%", "debt service coverage", "2.0"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    },
    {
        "input": "Update customer contact information and address for account holder 22334",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["not found in system", "Customer ID 22334 not found in system.", "not found"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    }
])
