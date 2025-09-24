from typing import Optional
from datetime import datetime
from langchain.tools import tool


def _score_dti_ratio(dti_ratio: float) -> int:
    """Score based on debt-to-income ratio."""
    if dti_ratio <= 28:
        return 25
    elif dti_ratio <= 36:
        return 20
    elif dti_ratio <= 43:
        return 15
    else:
        return 5


def _score_credit_score(credit_score: int) -> int:
    """Score based on credit score."""
    if credit_score >= 750:
        return 25
    elif credit_score >= 700:
        return 20
    elif credit_score >= 650:
        return 15
    elif credit_score >= 600:
        return 10
    else:
        return 5


def _score_loan_amount(loan_amount: float, monthly_income: float) -> int:
    """Score based on loan amount relative to income."""
    if loan_amount <= monthly_income * 12:
        return 25
    elif loan_amount <= monthly_income * 18:
        return 20
    elif loan_amount <= monthly_income * 24:
        return 15
    else:
        return 10


def _classify_risk(risk_score: int) -> tuple[str, str]:
    """Classify risk level and recommendation based on score."""
    if risk_score >= 70:
        return "LOW RISK", "APPROVE with standard terms"
    elif risk_score >= 50:
        return "MEDIUM RISK", "APPROVE with enhanced monitoring"
    elif risk_score >= 30:
        return "HIGH RISK", "REQUIRES additional documentation"
    else:
        return "VERY HIGH RISK", "RECOMMEND DENIAL"


def _get_dti_description(dti_ratio: float) -> str:
    """Get description for DTI ratio."""
    if dti_ratio <= 28:
        return "excellent"
    elif dti_ratio <= 36:
        return "good"
    elif dti_ratio <= 43:
        return "acceptable"
    else:
        return "concerning"


def _get_credit_description(credit_score: int) -> str:
    """Get description for credit score."""
    if credit_score >= 750:
        return "excellent"
    elif credit_score >= 700:
        return "good"
    elif credit_score >= 650:
        return "fair"
    else:
        return "poor"


# Credit Risk Analyzer Tool
@tool
def credit_risk_analyzer(
    customer_income: float,
    customer_debt: float,
    credit_score: int,
    loan_amount: float,
    loan_type: str = "personal"
) -> str:
    """
    Analyze credit risk for loan applications and credit decisions.

    This tool evaluates:
    - Debt-to-income ratio analysis
    - Credit score assessment
    - Loan-to-value calculations
    - Risk scoring and recommendations
    - Regulatory compliance checks

    Args:
        customer_income (float): Annual income in USD
        customer_debt (float): Total monthly debt payments in USD
        credit_score (int): FICO credit score (300-850)
        loan_amount (float): Requested loan amount in USD
        loan_type (str): Type of loan (personal, mortgage, business, auto)

    Returns:
        str: Comprehensive credit risk analysis and recommendations

    Examples:
        - "Analyze credit risk for $50k personal loan"
        - "Assess mortgage eligibility for $300k home purchase"
        - "Calculate risk score for business loan application"
    """
    # Calculate debt-to-income ratio
    monthly_income = customer_income / 12
    dti_ratio = (customer_debt / monthly_income) * 100

    # Calculate risk score using helper functions
    risk_score = (_score_dti_ratio(dti_ratio) +
                  _score_credit_score(credit_score) +
                  _score_loan_amount(loan_amount, monthly_income))

    # Get risk classification
    risk_level, recommendation = _classify_risk(risk_score)

    return f"""CREDIT RISK ANALYSIS REPORT
    ================================

    Customer Profile:
    - Annual Income: ${customer_income:,.2f}
    - Monthly Debt: ${customer_debt:,.2f}
    - Credit Score: {credit_score}
    - Loan Request: ${loan_amount:,.2f} ({loan_type})

    Risk Assessment:
    - Debt-to-Income Ratio: {dti_ratio:.1f}%
    - Risk Score: {risk_score}/75
    - Risk Level: {risk_level}

    Recommendation: {recommendation}

    Additional Notes:
    - DTI ratio of {dti_ratio:.1f}% is {_get_dti_description(dti_ratio)}
    - Credit score of {credit_score} is {_get_credit_description(credit_score)}
    - Loan amount represents {((loan_amount / customer_income) * 100):.1f}% of annual income
    """


def _get_customer_database():
    """Get mock customer database."""
    return {
        "12345": {
            "name": "John Smith",
            "checking_balance": 2547.89,
            "savings_balance": 12500.00,
            "credit_score": 745,
            "account_age_days": 450
        },
        "67890": {
            "name": "Sarah Johnson",
            "checking_balance": 892.34,
            "savings_balance": 3500.00,
            "credit_score": 680,
            "account_age_days": 180
        },
        "11111": {
            "name": "Business Corp LLC",
            "checking_balance": 45000.00,
            "savings_balance": 150000.00,
            "credit_score": 720,
            "account_age_days": 730
        }
    }


def _handle_check_balance(customer, account_type, customer_id):
    """Handle balance check action."""
    if account_type == "checking":
        balance = customer["checking_balance"]
    elif account_type == "savings":
        balance = customer["savings_balance"]
    else:
        return f"Account type '{account_type}' not supported for balance check."

    return f"""ACCOUNT BALANCE REPORT
    ================================

    Customer: {customer['name']}
    Account Type: {account_type.title()}
    Account ID: {customer_id}

    Current Balance: ${balance:,.2f}
    Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Account Status: Active
    """


def _handle_process_transaction(customer, account_type, amount, customer_id):
    """Handle transaction processing action."""
    if amount is None:
        return "Amount is required for transaction processing."

    if account_type == "checking":
        current_balance = customer["checking_balance"]
        if amount > 0:  # Deposit
            new_balance = current_balance + amount
            transaction_type = "DEPOSIT"
        else:  # Withdrawal
            if abs(amount) > current_balance:
                return f"Insufficient funds. Available balance: ${current_balance:,.2f}"
            new_balance = current_balance + amount  # amount is negative
            transaction_type = "WITHDRAWAL"

        # Update mock database
        customer["checking_balance"] = new_balance

    return f"""TRANSACTION PROCESSED
    ================================

    Customer: {customer['name']}
    Account: {account_type.title()} - {customer_id}
    Transaction: {transaction_type}
    Amount: ${abs(amount):,.2f}

    Previous Balance: ${current_balance:,.2f}
    New Balance: ${new_balance:,.2f}
    Transaction ID: TX{datetime.now().strftime('%Y%m%d%H%M%S')}

    Status: Completed
    """


def _get_product_recommendations(credit_score):
    """Get product recommendations based on credit score."""
    if credit_score >= 700:
        return [
            "Premium Checking Account with no monthly fees",
            "High-Yield Savings Account (2.5% APY)",
            "Personal Line of Credit up to $25,000",
            "Investment Advisory Services"
        ]
    elif credit_score >= 650:
        return [
            "Standard Checking Account",
            "Basic Savings Account (1.2% APY)",
            "Secured Credit Card",
            "Debt Consolidation Loan"
        ]
    else:
        return [
            "Second Chance Checking Account",
            "Basic Savings Account (0.5% APY)",
            "Secured Credit Card",
            "Credit Building Services"
        ]


def _handle_recommend_product(customer):
    """Handle product recommendation action."""
    recommendations = _get_product_recommendations(customer["credit_score"])

    return f"""PRODUCT RECOMMENDATIONS
    ================================

    Customer: {customer['name']}
    Credit Score: {customer['credit_score']}
    Account Age: {customer['account_age_days']} days

    Recommended Products:
    {chr(10).join(f"  • {rec}" for rec in recommendations)}

    Next Steps:
    - Schedule consultation with relationship manager
    - Review product terms and conditions
    - Complete application process
    """


def _handle_get_info(customer, customer_id):
    """Handle get info action."""
    credit_tier = ('Excellent' if customer['credit_score'] >= 750 else 
                   'Good' if customer['credit_score'] >= 700 else 
                   'Fair' if customer['credit_score'] >= 650 else 'Poor')

    return f"""CUSTOMER ACCOUNT INFORMATION
    ================================

    Customer ID: {customer_id}
    Name: {customer['name']}
    Account Age: {customer['account_age_days']} days

    Account Balances:
    - Checking: ${customer['checking_balance']:,.2f}
    - Savings: {customer['savings_balance']:,.2f}

    Credit Profile:
    - Credit Score: {customer['credit_score']}
    - Credit Tier: {credit_tier}

    Services Available:
    - Online Banking
    - Mobile App
    - Bill Pay
    - Direct Deposit
    """


# Customer Account Manager Tool
@tool
def customer_account_manager(
    account_type: str,
    customer_id: str,
    action: str,
    amount: Optional[float] = None,
    account_details: Optional[str] = None
) -> str:
    """
    Manage customer accounts and provide banking services.

    This tool handles:
    - Account information and balances
    - Transaction processing
    - Product recommendations
    - Customer service inquiries
    - Account maintenance

    Args:
        account_type (str): Type of account (checking, savings, loan, credit_card)
        customer_id (str): Customer identifier
        action (str): Action to perform (check_balance, process_transaction, recommend_product, get_info)
        amount (float, optional): Transaction amount for financial actions
        account_details (str, optional): Additional account information

    Returns:
        str: Account information or transaction results

    Examples:
        - "Check balance for checking account 12345"
        - "Process $500 deposit to savings account 67890"
        - "Recommend products for customer with high balance"
        - "Get account information for loan account 11111"
    """
    customer_db = _get_customer_database()

    if customer_id not in customer_db:
        return f"Customer ID {customer_id} not found in system."

    customer = customer_db[customer_id]

    if action == "check_balance":
        return _handle_check_balance(customer, account_type, customer_id)
    elif action == "process_transaction":
        return _handle_process_transaction(customer, account_type, amount, customer_id)
    elif action == "recommend_product":
        return _handle_recommend_product(customer)
    elif action == "get_info":
        return _handle_get_info(customer, customer_id)
    else:
        return f"Action '{action}' not supported. Available actions: check_balance, process_transaction, recommend_product, get_info"


# Fraud Detection System Tool
@tool
def fraud_detection_system(
    transaction_id: str,
    customer_id: str,
    transaction_amount: float,
    transaction_type: str,
    location: str,
    device_id: Optional[str] = None
) -> str:
    """
    Analyze transactions for potential fraud and security risks.

    This tool evaluates:
    - Transaction patterns and anomalies
    - Geographic risk assessment
    - Device fingerprinting
    - Behavioral analysis
    - Risk scoring and alerts

    Args:
        transaction_id (str): Unique transaction identifier
        customer_id (str): Customer account identifier
        transaction_amount (float): Transaction amount in USD
        transaction_type (str): Type of transaction (purchase, withdrawal, transfer, deposit)
        location (str): Transaction location or IP address
        device_id (str, optional): Device identifier for mobile/online transactions

    Returns:
        str: Fraud risk assessment and recommendations

    Examples:
        - "Analyze fraud risk for $500 ATM withdrawal in Miami"
        - "Check security for $2000 online purchase from new device"
        - "Assess risk for $10000 wire transfer to international account"
    """

    # Mock fraud detection logic
    risk_score = 0
    risk_factors = []
    recommendations = []

    # Amount-based risk
    if transaction_amount > 10000:
        risk_score += 30
        risk_factors.append("High-value transaction (>$10k)")
        recommendations.append("Require additional verification")

    if transaction_amount > 1000:
        risk_score += 15
        risk_factors.append("Medium-value transaction (>$1k)")

    # Location-based risk
    high_risk_locations = ["Nigeria", "Russia", "North Korea", "Iran", "Cuba"]
    if any(country in location for country in high_risk_locations):
        risk_score += 40
        risk_factors.append("High-risk geographic location")
        recommendations.append("Block transaction - high-risk country")

    # Transaction type risk
    if transaction_type == "withdrawal" and transaction_amount > 5000:
        risk_score += 25
        risk_factors.append("Large cash withdrawal")
        recommendations.append("Require in-person verification")

    if transaction_type == "transfer" and transaction_amount > 5000:
        risk_score += 20
        risk_factors.append("Large transfer")
        recommendations.append("Implement 24-hour delay for verification")

    # Device risk
    if device_id and device_id.startswith("UNKNOWN"):
        risk_score += 25
        risk_factors.append("Unknown or new device")
        recommendations.append("Require multi-factor authentication")

    # Time-based risk (mock: assume night transactions are riskier)
    current_hour = datetime.now().hour
    if 22 <= current_hour or current_hour <= 6:
        risk_score += 10
        risk_factors.append("Unusual transaction time")

    # Risk classification
    if risk_score >= 70:
        risk_level = "HIGH RISK"
        action = "BLOCK TRANSACTION"
    elif risk_score >= 40:
        risk_level = "MEDIUM RISK"
        action = "REQUIRE VERIFICATION"
    else:
        risk_level = "LOW RISK"
        action = "ALLOW TRANSACTION"

    return f"""FRAUD DETECTION ANALYSIS
    ================================

    Transaction Details:
    - Transaction ID: {transaction_id}
    - Customer ID: {customer_id}
    - Amount: ${transaction_amount:,.2f}
    - Type: {transaction_type.title()}
    - Location: {location}
    - Device: {device_id or 'N/A'}

    Risk Assessment: {risk_level}
    - Risk Score: {risk_score}/100
    - Risk Factors: {len(risk_factors)}

    Identified Risk Factors:
    {chr(10).join(f"  • {factor}" for factor in risk_factors)}

    Recommendations:
    {chr(10).join(f"  • {rec}" for rec in recommendations) if recommendations else "  • No additional actions required"}

    Decision: {action}

    Next Steps:
    - Log risk assessment in fraud monitoring system
    - Update customer risk profile if necessary
    - Monitor for similar patterns
    """


# Export all banking tools
AVAILABLE_TOOLS = [
    credit_risk_analyzer,
    customer_account_manager,
    fraud_detection_system
]

if __name__ == "__main__":
    print("Banking-specific tools created!")
    print(f"Available tools: {len(AVAILABLE_TOOLS)}")
    for banking_tool in AVAILABLE_TOOLS:
        print(f"   - {banking_tool.name}: {banking_tool.description[:80]}...")
