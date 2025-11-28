"""
SQL query templates to avoid LLM generating full queries.
LLM only provides parameters (dates, categories, limits).
"""

from typing import Optional, Dict, Any


class SQLTemplates:
    """Pre-defined SQL templates with parameter slots."""
    
    @staticmethod
    def top_products_revenue(limit_n: int = 3, start_date: str = None, end_date: str = None) -> str:
        """Top N products by revenue."""
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND o.OrderDate >= '{start_date}' AND o.OrderDate <= '{end_date}'"
        elif start_date:
            date_filter = f"AND o.OrderDate >= '{start_date}'"
        
        return f"""
        SELECT 
            p.ProductName as product,
            SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue
        FROM "Order Details" od
        JOIN Products p ON od.ProductID = p.ProductID
        JOIN Orders o ON od.OrderID = o.OrderID
        WHERE 1=1 {date_filter}
        GROUP BY p.ProductName
        ORDER BY revenue DESC
        LIMIT {limit_n}
        """.strip()
    
    @staticmethod
    def category_quantity(category: str = None, start_date: str = None, end_date: str = None) -> str:
        """Total quantity by category (returns all categories if category is None)."""
        category_filter = ""
        if category and category != "ALL":
            category_filter = f"AND c.CategoryName = '{category}'"
        
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND o.OrderDate >= '{start_date}' AND o.OrderDate <= '{end_date}'"
        elif start_date:
            date_filter = f"AND o.OrderDate >= '{start_date}'"
        
        return f"""
        SELECT 
            c.CategoryName as category,
            SUM(od.Quantity) as quantity
        FROM "Order Details" od
        JOIN Products p ON od.ProductID = p.ProductID
        JOIN Categories c ON p.CategoryID = c.CategoryID
        JOIN Orders o ON od.OrderID = o.OrderID
        WHERE 1=1 {category_filter} {date_filter}
        GROUP BY c.CategoryName
        ORDER BY quantity DESC
        """.strip()
    
    @staticmethod
    def aov(start_date: str = None, end_date: str = None) -> str:
        """Average Order Value."""
        date_filter = ""
        if start_date and end_date:
            date_filter = f"WHERE o.OrderDate >= '{start_date}' AND o.OrderDate <= '{end_date}'"
        elif start_date:
            date_filter = f"WHERE o.OrderDate >= '{start_date}'"
        
        return f"""
        SELECT 
            SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID) as aov
        FROM "Order Details" od
        JOIN Orders o ON od.OrderID = o.OrderID
        {date_filter}
        """
    
    @staticmethod
    def revenue_by_category(category: str, start_date: str = None, end_date: str = None) -> str:
        """Revenue for a specific category."""
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND o.OrderDate >= '{start_date}' AND o.OrderDate <= '{end_date}'"
        elif start_date:
            date_filter = f"AND o.OrderDate >= '{start_date}'"
        
        return f"""
        SELECT 
            SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue
        FROM "Order Details" od
        JOIN Products p ON od.ProductID = p.ProductID
        JOIN Categories c ON p.CategoryID = c.CategoryID
        JOIN Orders o ON od.OrderID = o.OrderID
        WHERE c.CategoryName = '{category}' {date_filter}
        """.strip()
    
    @staticmethod
    def customer_margin(start_date: str = None, end_date: str = None, cost_multiplier: float = 0.7) -> str:
        """Customer gross margin (approximating cost)."""
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND o.OrderDate >= '{start_date}' AND o.OrderDate <= '{end_date}'"
        elif start_date:
            date_filter = f"AND o.OrderDate >= '{start_date}'"
        
        return f"""
        SELECT 
            cu.CompanyName as customer,
            SUM((od.UnitPrice - (od.UnitPrice * {cost_multiplier})) * od.Quantity * (1 - od.Discount)) as margin
        FROM "Order Details" od
        JOIN Orders o ON od.OrderID = o.OrderID
        JOIN Customers cu ON o.CustomerID = cu.CustomerID
        WHERE 1=1 {date_filter}
        GROUP BY cu.CompanyName
        ORDER BY margin DESC
        LIMIT 1
        """.strip()
    
    @staticmethod
    def build_query(query_type: str, params: Dict[str, Any]) -> str:
        """Build SQL from template and parameters."""
        if query_type == "top_products_revenue":
            return SQLTemplates.top_products_revenue(
                limit_n=int(params.get('limit_n', 3)),
                start_date=params.get('filter_start_date'),
                end_date=params.get('filter_end_date')
            )
        elif query_type == "category_quantity":
            return SQLTemplates.category_quantity(
                category=params.get('filter_category') if params.get('filter_category') != 'ALL' else None,
                start_date=params.get('filter_start_date'),
                end_date=params.get('filter_end_date')
            )
        elif query_type == "aov":
            return SQLTemplates.aov(
                start_date=params.get('filter_start_date'),
                end_date=params.get('filter_end_date')
            )
        elif query_type == "revenue_by_category":
            return SQLTemplates.revenue_by_category(
                category=params.get('filter_category'),
                start_date=params.get('filter_start_date'),
                end_date=params.get('filter_end_date')
            )
        elif query_type == "customer_margin":
            return SQLTemplates.customer_margin(
                start_date=params.get('filter_start_date'),
                end_date=params.get('filter_end_date')
            )
        else:
            raise ValueError(f"Unknown query type: {query_type}")