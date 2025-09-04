import pandas as pd
import uuid

# Banking-specific test dataset for retail and commercial banking
banking_test_dataset = pd.DataFrame([
    {
        "input": "Analyze credit risk for a $50,000 personal loan application with $75,000 annual income, $1,200 monthly debt, and 720 credit score",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["LOW RISK", "MEDIUM RISK", "APPROVE", "debt-to-income ratio", "risk score"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    },
    {
        "input": "Check SR 11-7 compliance for a $150,000 business loan to a commercial customer with 90-day old account",
        "expected_tools": ["compliance_monitor"],
        "possible_outputs": ["SR 11-7", "model validation", "compliance", "risk level", "required actions"],
        "session_id": str(uuid.uuid4()),
        "category": "compliance"
    },
    {
        "input": "Calculate monthly payment for a $300,000 mortgage at 4.5% interest for 30 years",
        "expected_tools": ["financial_calculator"],
        "possible_outputs": ["monthly payment", "amortization", "total interest", "loan payment calculation"],
        "session_id": str(uuid.uuid4()),
        "category": "financial_calculation"
    },
    {
        "input": "Check account balance for checking account 12345",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["balance", "account information", "John Smith", "checking account"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Analyze fraud risk for a $15,000 wire transfer from customer 67890 to Nigeria",
        "expected_tools": ["fraud_detection_system"],
        "possible_outputs": ["HIGH RISK", "fraud detection", "risk score", "geographic risk", "block transaction"],
        "session_id": str(uuid.uuid4()),
        "category": "fraud_detection"
    },
    {
        "input": "Verify AML compliance for a $25,000 deposit from a new customer account opened 15 days ago",
        "expected_tools": ["compliance_monitor"],
        "possible_outputs": ["KYC/AML", "enhanced due diligence", "CTR filing", "compliance issues"],
        "session_id": str(uuid.uuid4()),
        "category": "compliance"
    },
    {
        "input": "Recommend banking products for customer 11111 with $150,000 in savings and 720 credit score",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["product recommendations", "premium accounts", "investment services", "line of credit"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Calculate investment growth for $100,000 at 8% annual return over 10 years",
        "expected_tools": ["financial_calculator"],
        "possible_outputs": ["future value", "total return", "annualized return", "investment growth"],
        "session_id": str(uuid.uuid4()),
        "category": "financial_calculation"
    },
    {
        "input": "Assess credit risk for a $1,000,000 commercial real estate loan with $500,000 annual business income",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["HIGH RISK", "VERY HIGH RISK", "business loan", "commercial", "risk assessment"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    },
    {
        "input": "Process a $2,500 deposit to savings account 67890",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["transaction processed", "deposit", "new balance", "transaction ID"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    }
])

print("Banking-specific test dataset created!")
print(f"Number of test cases: {len(banking_test_dataset)}")
print(f"Categories: {banking_test_dataset['category'].unique()}")
print(f"Tools being tested: {sorted(banking_test_dataset['expected_tools'].explode().unique())}")

# Display sample test cases
print("\nSample test cases:")
for i, row in banking_test_dataset.head(3).iterrows():
    print(f"{i+1}. {row['input'][:80]}... -> Expected tool: {row['expected_tools'][0]} ({row['category']})")
