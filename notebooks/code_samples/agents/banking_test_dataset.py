import pandas as pd
import uuid

# Banking-specific test dataset for retail and commercial banking
# tools: credit_risk_analyzer, customer_account_manager, fraud_detection_system
banking_test_dataset = pd.DataFrame([
    {
        "input": "Analyze credit risk for a $50,000 personal loan application with $75,000 annual income, $1,200 monthly debt, and 720 credit score",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["LOW RISK", "MEDIUM RISK", "APPROVE", "debt-to-income ratio", "19.2%", "risk score", "720", "probability of default", "2.5%"],
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
        "possible_outputs": ["balance", "$3,247.82", "account information", "John Smith", "checking account", "available balance", "$3,047.82"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Analyze fraud risk for a $15,000 wire transfer from customer 67890 to Nigeria",
        "expected_tools": ["fraud_detection_system"],
        "possible_outputs": ["HIGH RISK", "fraud score", "87", "geographic risk", "95%", "amount", "$15,000", "block transaction", "confidence", "92%"],
        "session_id": str(uuid.uuid4()),
        "category": "fraud_detection"
    },
    {
        "input": "Recommend banking products for customer 11111 with $150,000 in savings and 720 credit score",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["product recommendations", "premium accounts", "investment services", "line of credit", "$50,000", "savings rate", "4.25%"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Investigate suspicious transactions totaling $75,000 across multiple accounts in the last week",
        "expected_tools": ["fraud_detection_system"],
        "possible_outputs": ["suspicious activity", "pattern analysis", "transaction monitoring", "VERY HIGH RISK", "alert", "fraud score", "94", "total amount", "$75,000"],
        "session_id": str(uuid.uuid4()),
        "category": "fraud_detection"
    },
    {
        "input": "Assess credit risk for a $1,000,000 commercial real estate loan with $500,000 annual business income",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["HIGH RISK", "VERY HIGH RISK", "business loan", "commercial", "risk assessment", "loan-to-value", "66.7%", "debt service coverage", "2.0"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    },
    {
        "input": "Process a $2,500 deposit to savings account 67890",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["transaction processed", "deposit", "$2,500", "new balance", "$15,847.32", "transaction ID", "TXN-789456123"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Review credit card application for customer with 580 credit score, $42,000 annual income, and recent bankruptcy",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["VERY HIGH RISK", "DECLINE", "bankruptcy", "credit score", "580", "probability of default", "35%", "debt-to-income", "78%"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    },
    {
        "input": "Update customer contact information and address for account holder 22334",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["customer updated", "address change", "contact information", "profile updated", "customer ID", "22334"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Detect potential fraud in multiple small transactions under $500 happening rapidly from different locations",
        "expected_tools": ["fraud_detection_system"],
        "possible_outputs": ["velocity fraud", "geographic anomaly", "HIGH RISK", "transaction pattern", "card fraud", "velocity score", "89", "locations", "4"],
        "session_id": str(uuid.uuid4()),
        "category": "fraud_detection"
    },
    {
        "input": "Close dormant account 98765 and transfer remaining balance to active checking account",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["account closed", "balance transfer", "$487.63", "dormant account", "transaction completed", "account ID", "98765"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Assess credit risk for auto loan of $35,000 for customer with 650 credit score, $55,000 income, and no previous auto loans",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["MEDIUM RISK", "auto loan", "first-time borrower", "acceptable risk", "interest rate", "6.75%", "monthly payment", "$574"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    },
    {
        "input": "Flag unusual ATM withdrawals of $500 every hour for the past 6 hours from account 44556",
        "expected_tools": ["fraud_detection_system"],
        "possible_outputs": ["velocity pattern", "ATM fraud", "HIGH RISK", "card compromise", "unusual pattern", "total withdrawn", "$3,000", "frequency", "6", "transactions"],
        "session_id": str(uuid.uuid4()),
        "category": "fraud_detection"
    },
    {
        "input": "Open new business checking account for LLC with $25,000 initial deposit and setup online banking",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["business account", "new account", "online banking setup", "LLC registration", "account opened", "initial deposit", "$25,000", "account number", "987654321"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Evaluate creditworthiness for student loan refinancing of $85,000 with recent graduation and $65,000 starting salary",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["student loan", "refinancing", "MEDIUM RISK", "recent graduate", "debt consolidation", "new rate", "4.5%", "monthly payment", "$878"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    },
    {
        "input": "Investigate merchant transactions showing unusual chargeback patterns and potential money laundering",
        "expected_tools": ["fraud_detection_system"],
        "possible_outputs": ["merchant fraud", "chargeback analysis", "money laundering", "VERY HIGH RISK", "compliance alert", "chargeback rate", "15.3%", "risk score", "96"],
        "session_id": str(uuid.uuid4()),
        "category": "fraud_detection"
    },
    {
        "input": "Set up automatic bill pay for customer 77889 for utilities, mortgage, and insurance payments",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["automatic payments", "bill pay setup", "recurring transactions", "payment scheduling", "total monthly", "$2,847", "customer ID", "77889"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Analyze credit risk for line of credit increase from $10,000 to $25,000 for existing customer with payment history",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["credit limit increase", "LOW RISK", "payment history", "existing customer", "new limit", "$25,000", "utilization", "12%"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    },
    {
        "input": "Review suspicious cryptocurrency exchange transactions totaling $200,000 over 3 days from business account",
        "expected_tools": ["fraud_detection_system"],
        "possible_outputs": ["cryptocurrency", "large transactions", "business account", "HIGH RISK", "regulatory concern", "total amount", "$200,000", "risk score", "91"],
        "session_id": str(uuid.uuid4()),
        "category": "fraud_detection"
    },
    {
        "input": "Process stop payment request for check #1234 and issue new checks for customer account 55667",
        "expected_tools": ["customer_account_manager"],
        "possible_outputs": ["stop payment", "check services", "new checks", "payment blocked", "customer service", "check amount", "$1,247.50", "account", "55667"],
        "session_id": str(uuid.uuid4()),
        "category": "account_management"
    },
    {
        "input": "Evaluate mortgage pre-approval for $450,000 home purchase with 20% down payment, 780 credit score, and $125,000 household income",
        "expected_tools": ["credit_risk_analyzer"],
        "possible_outputs": ["mortgage pre-approval", "LOW RISK", "excellent credit", "strong income", "home purchase", "approved amount", "$450,000", "interest rate", "3.75%", "monthly payment", "$2,083"],
        "session_id": str(uuid.uuid4()),
        "category": "credit_risk"
    }
])
