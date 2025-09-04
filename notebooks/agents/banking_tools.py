from typing import Optional
from datetime import datetime
from langchain.tools import tool

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
    
    # Risk scoring based on multiple factors
    risk_score = 0
    
    # DTI ratio scoring
    if dti_ratio <= 28:
        risk_score += 25
    elif dti_ratio <= 36:
        risk_score += 20
    elif dti_ratio <= 43:
        risk_score += 15
    else:
        risk_score += 5
    
    # Credit score scoring
    if credit_score >= 750:
        risk_score += 25
    elif credit_score >= 700:
        risk_score += 20
    elif credit_score >= 650:
        risk_score += 15
    elif credit_score >= 600:
        risk_score += 10
    else:
        risk_score += 5
    
    # Loan amount scoring
    if loan_amount <= monthly_income * 12:
        risk_score += 25
    elif loan_amount <= monthly_income * 18:
        risk_score += 20
    elif loan_amount <= monthly_income * 24:
        risk_score += 15
    else:
        risk_score += 10
    
    # Risk classification
    if risk_score >= 70:
        risk_level = "LOW RISK"
        recommendation = "APPROVE with standard terms"
    elif risk_score >= 50:
        risk_level = "MEDIUM RISK"
        recommendation = "APPROVE with enhanced monitoring"
    elif risk_score >= 30:
        risk_level = "HIGH RISK"
        recommendation = "REQUIRES additional documentation"
    else:
        risk_level = "VERY HIGH RISK"
        recommendation = "RECOMMEND DENIAL"
    
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
    - DTI ratio of {dti_ratio:.1f}% is {'excellent' if dti_ratio <= 28 else 'good' if dti_ratio <= 36 else 'acceptable' if dti_ratio <= 43 else 'concerning'}
    - Credit score of {credit_score} is {'excellent' if credit_score >= 750 else 'good' if credit_score >= 700 else 'fair' if credit_score >= 650 else 'poor'}
    - Loan amount represents {((loan_amount / customer_income) * 100):.1f}% of annual income
    """

# Compliance Monitor Tool
@tool
def compliance_monitor(
    transaction_type: str,
    transaction_amount: float,
    customer_type: str,
    account_age_days: int,
    compliance_framework: str = "SR_11_7"
) -> str:
    """
    Monitor transactions and operations for regulatory compliance.
    
    This tool checks compliance with:
    - SR 11-7: Supervisory Guidance on Model Risk Management
    - SS 1-23: Supervisory Guidance on Model Risk Management
    - KYC/AML requirements
    - Transaction monitoring rules
    - Regulatory reporting requirements
    
    Args:
        transaction_type (str): Type of transaction (deposit, withdrawal, transfer, loan)
        transaction_amount (float): Transaction amount in USD
        customer_type (str): Customer classification (retail, commercial, high_net_worth)
        account_age_days (int): Age of account in days
        compliance_framework (str): Compliance framework to check (SR_11_7, SS_1_23, KYC_AML)
    
    Returns:
        str: Compliance assessment and required actions
        
    Examples:
        - "Check SR 11-7 compliance for $100k business loan"
        - "Verify AML compliance for $25k wire transfer"
        - "Assess model risk for new credit scoring algorithm"
    """
    
    compliance_issues = []
    required_actions = []
    risk_level = "LOW"
    
    # SR 11-7 Model Risk Management checks
    if compliance_framework in ["SR_11_7", "SS_1_23"]:
        if transaction_amount > 100000:
            compliance_issues.append("Large transaction requires enhanced model validation per SR 11-7")
            required_actions.append("Implement additional model monitoring and validation")
            risk_level = "MEDIUM"
        
        if customer_type == "commercial" and transaction_amount > 50000:
            compliance_issues.append("Commercial transaction requires business model validation")
            required_actions.append("Document business model assumptions and limitations")
            risk_level = "MEDIUM"
    
    # KYC/AML compliance checks
    if compliance_framework == "KYC_AML":
        if transaction_amount > 10000:
            compliance_issues.append("Transaction above $10k requires CTR filing")
            required_actions.append("File Currency Transaction Report (CTR)")
            risk_level = "MEDIUM"
        
        if account_age_days < 30 and transaction_amount > 5000:
            compliance_issues.append("New account with significant transaction requires enhanced due diligence")
            required_actions.append("Conduct enhanced customer due diligence")
            risk_level = "HIGH"
    
    # General compliance checks
    if transaction_amount > 1000000:
        compliance_issues.append("Million-dollar transaction requires senior management approval")
        required_actions.append("Obtain senior management approval and document decision")
        risk_level = "HIGH"
    
    if not compliance_issues:
        compliance_issues.append("No compliance issues detected")
        required_actions.append("Standard monitoring procedures apply")
    
    return f"""COMPLIANCE MONITORING REPORT
    ================================
    
    Transaction Details:
    - Type: {transaction_type.title()}
    - Amount: ${transaction_amount:,.2f}
    - Customer Type: {customer_type.replace('_', ' ').title()}
    - Account Age: {account_age_days} days
    - Framework: {compliance_framework.replace('_', ' ').title()}
    
    Compliance Assessment:
    - Risk Level: {risk_level}
    - Issues Found: {len(compliance_issues)}
    
    Compliance Issues:
    {chr(10).join(f"  • {issue}" for issue in compliance_issues)}
    
    Required Actions:
    {chr(10).join(f"  • {action}" for action in required_actions)}
    
    Regulatory References:
    - SR 11-7: Model Risk Management
    - SS 1-23: Model Risk Management
    - KYC/AML: Customer Due Diligence
    """

# Financial Calculator Tool
@tool
def financial_calculator(
    calculation_type: str,
    principal: float,
    rate: float,
    term: int,
    payment_frequency: str = "monthly"
) -> str:
    """
    Perform banking-specific financial calculations.
    
    This tool calculates:
    - Loan payments and amortization
    - Interest calculations
    - Investment returns
    - Account balances
    - Financial ratios
    
    Args:
        calculation_type (str): Type of calculation (loan_payment, interest, investment, balance)
        principal (float): Principal amount in USD
        rate (float): Annual interest rate as percentage
        term (int): Term in years or months
        payment_frequency (str): Payment frequency (monthly, quarterly, annually)
    
    Returns:
        str: Detailed calculation results and breakdown
        
    Examples:
        - "Calculate monthly payment for $200k mortgage at 4.5% for 30 years"
        - "Compute interest earned on $10k savings at 2.5% for 5 years"
        - "Determine investment growth for $50k at 8% return over 10 years"
    """
    
    # Convert annual rate to periodic rate
    if payment_frequency == "monthly":
        periodic_rate = rate / 100 / 12
        periods = term * 12
    elif payment_frequency == "quarterly":
        periodic_rate = rate / 100 / 4
        periods = term * 4
    else:  # annually
        periodic_rate = rate / 100
        periods = term
    
    if calculation_type == "loan_payment":
        if periodic_rate == 0:
            monthly_payment = principal / periods
        else:
            monthly_payment = principal * (periodic_rate * (1 + periodic_rate)**periods) / ((1 + periodic_rate)**periods - 1)
        
        total_payments = monthly_payment * periods
        total_interest = total_payments - principal
        
        return f"""LOAN PAYMENT CALCULATION
        ================================
        
        Loan Details:
        - Principal: ${principal:,.2f}
        - Annual Rate: {rate:.2f}%
        - Term: {term} years ({periods} {payment_frequency} payments)
        - Payment Frequency: {payment_frequency.title()}
        
        Results:
        - {payment_frequency.title()} Payment: ${monthly_payment:,.2f}
        - Total Payments: ${total_payments:,.2f}
        - Total Interest: ${total_interest:,.2f}
        - Interest Percentage: {((total_interest / total_payments) * 100):.1f}%
        """
        
    elif calculation_type == "interest":
        simple_interest = principal * (rate / 100) * term
        compound_interest = principal * ((1 + rate / 100) ** term - 1)
        
        return f"""INTEREST CALCULATION
        ================================
        
        Investment Details:
        - Principal: ${principal:,.2f}
        - Annual Rate: {rate:.2f}%
        - Term: {term} years
        
        Results:
        - Simple Interest: ${simple_interest:,.2f}
        - Compound Interest: ${compound_interest:,.2f}
        - Final Amount (Simple): ${principal + simple_interest:,.2f}
        - Final Amount (Compound): ${principal + compound_interest:,.2f}
        - Interest Difference: ${compound_interest - simple_interest:,.2f}
        """
    
    elif calculation_type == "investment":
        future_value = principal * ((1 + rate / 100) ** term)
        total_return = future_value - principal
        annualized_return = ((future_value / principal) ** (1 / term) - 1) * 100
        
        return f"""INVESTMENT GROWTH CALCULATION
        ================================
        
        Investment Details:
        - Initial Investment: ${principal:,.2f}
        - Annual Return: {rate:.2f}%
        - Time Period: {term} years
        
        Results:
        - Future Value: ${future_value:,.2f}
        - Total Return: ${total_return:,.2f}
        - Annualized Return: {annualized_return:.2f}%
        - Growth Multiple: {future_value / principal:.2f}x
        """
    
    else:
        return f"Calculation type '{calculation_type}' not supported. Available types: loan_payment, interest, investment"

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
    
    # Mock customer database
    customer_db = {
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
    
    if customer_id not in customer_db:
        return f"Customer ID {customer_id} not found in system."
    
    customer = customer_db[customer_id]
    
    if action == "check_balance":
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
    
    elif action == "process_transaction":
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
    
    elif action == "recommend_product":
        if customer["credit_score"] >= 700:
            recommendations = [
                "Premium Checking Account with no monthly fees",
                "High-Yield Savings Account (2.5% APY)",
                "Personal Line of Credit up to $25,000",
                "Investment Advisory Services"
            ]
        elif customer["credit_score"] >= 650:
            recommendations = [
                "Standard Checking Account",
                "Basic Savings Account (1.2% APY)",
                "Secured Credit Card",
                "Debt Consolidation Loan"
            ]
        else:
            recommendations = [
                "Second Chance Checking Account",
                "Basic Savings Account (0.5% APY)",
                "Secured Credit Card",
                "Credit Building Services"
            ]
        
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
    
    elif action == "get_info":
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
        - Credit Tier: {'Excellent' if customer['credit_score'] >= 750 else 'Good' if customer['credit_score'] >= 700 else 'Fair' if customer['credit_score'] >= 650 else 'Poor'}
        
        Services Available:
        - Online Banking
        - Mobile App
        - Bill Pay
        - Direct Deposit
        """
    
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
        color = "🔴"
    elif risk_score >= 40:
        risk_level = "MEDIUM RISK"
        action = "REQUIRE VERIFICATION"
        color = "🟡"
    else:
        risk_level = "LOW RISK"
        action = "ALLOW TRANSACTION"
        color = "🟢"
    
    return f"""FRAUD DETECTION ANALYSIS
    ================================
    
    Transaction Details:
    - Transaction ID: {transaction_id}
    - Customer ID: {customer_id}
    - Amount: ${transaction_amount:,.2f}
    - Type: {transaction_type.title()}
    - Location: {location}
    - Device: {device_id or 'N/A'}
    
    Risk Assessment: {color} {risk_level}
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
    compliance_monitor,
    financial_calculator,
    customer_account_manager,
    fraud_detection_system
]

if __name__ == "__main__":
    print("Banking-specific tools created!")
    print(f"Available tools: {len(AVAILABLE_TOOLS)}")
    for tool in AVAILABLE_TOOLS:
        print(f"   - {tool.name}: {tool.description[:80]}...")
