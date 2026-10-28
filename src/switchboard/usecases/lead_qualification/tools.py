from datetime import date, timedelta
from typing import Literal

from langchain_core.tools import BaseTool, tool
from sqlalchemy import func, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


from switchboard.usecases.lead_qualification.db import Company, Contact, Task

WRITE_TOOLS = {"create_contact", "update_contact", "create_task"}

MOCK_ENRICHMENT = {
    "initech.example": {"name": "Initech", "industry": "Software", "employee_count": 300},
    "umbrella.example": {"name": "Umbrella Health", "industry": "Healthcare", "employee_count": 5000},
    "pied-piper.example": {"name": "Pied Piper", "industry": "Software", "employee_count": 12},
}

def _contact_dict(c: Contact) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "email": c.email,
        "title": c.title,
        "status": c.status,
        "company": c.company.name if c.company else None,
    }


def make_tools(engine: Engine) -> list[BaseTool]:

    @tool(parse_docstring=True)
    def search_contacts(query: str) -> list[dict]:
         
         """Search CRM contacts by exact email, part of a name, or company domain.

        Args:
            query: An email address, a name fragment, or a domain like 'acme.example'.
        """
         q = query.strip().lower()
         with Session(engine) as s:
             stmt = (
                select(Contact)
                .outerjoin(Company)
                .where(
                    or_(
                        func.lower(Contact.email) == q,
                        Contact.name.ilike(f"%{q}%"),
                        Company.domain ==q,
                    )
                )
                .limit(10)
             )
             contacts = [_contact_dict(c) for c in s.scalars(stmt)]
         return {"count" : len(contacts), "contacts": contacts}

    @tool(parse_docstring=True)
    def lookup_company(domain:str) -> dict:
        """
        Get company info for a domain: CRM data and open deals if known otherwise enrichment data

        Args:
            domain: The company's web domain, eg. 'acmerobotics.example'.    
        """
        domain = domain.strip().lower()
        with Session(engine) as s:
            company = s.scalar(
                select(Company)
                .where(Company.domain == domain)               
            )

            if company:
                return{
                    "found" : True,
                    "source" : "crm",
                    "name" : company.name,
                    "industry" : company.industry,
                    "employee_count" : company.employee_count,
                    "contact_count" : len(company.contacts),
                    "open_deals" : [
                        {
                            "name" : d.name, "amount" : d.amount
                        } for d in company.deals if d.stage == "open"
                    ],
                }
        if domain in MOCK_ENRICHMENT:
            return {"found": True, "source": "enrichment", **MOCK_ENRICHMENT[domain]}
        return {"found": False, "domain": domain}


    @tool(parse_docstring=True)
    def create_contact(
        name: str,
        email: str,
        title: str | None = None,
        company_domain: str | None = None,
        company_name: str | None = None,
    ) -> dict:
        """Create a new CRM contact. Fails if the email already exists (use update_contact instead).

        Args:
            name: Full name of the person.
            email: Their work email address.
            title: Job title, if known.
            company_domain: Company web domain. The company is created if it is not in the CRM yet.
            company_name: Company name, used only when a new company has to be created.
        """
        email = email.strip().lower()
        with Session(engine) as s:
            if s.scalar(select(Contact).where(Contact.email == email)):
                return {"error": f"Contact {email} already exists. Use update_contact."}

            company = None
            if company_domain:
                company_domain = company_domain.strip().lower()
                company = s.scalar(select(Company).where(Company.domain == company_domain))
                if company is None:
                    company = Company(name=company_name or company_domain, domain=company_domain)

            contact = Contact(name=name, email=email, title=title, company=company)
            s.add(contact)
            s.commit()
            return {"created": _contact_dict(contact)}

    @tool(parse_docstring=True)
    def update_contact(
        email: str,
        title: str | None = None,
        status: Literal["new", "qualified", "disqualified"] | None = None,
    ) -> dict:
        """Update an existing CRM contact, found by email. Only the given fields change.

        Args:
            email: Email of the contact to update.
            title: New job title.
            status: New lead status.
        """
        if title is None and status is None:
            return {"error": "Nothing to update: pass title and/or status."}
        email = email.strip().lower()
        with Session(engine) as s:
            contact = s.scalar(select(Contact).where(Contact.email == email))
            if contact is None:
                return {"error": f"No contact with email {email}. Use create_contact."}
            if title is not None:
                contact.title = title
            if status is not None:
                contact.status = status
            s.commit()
            return {"updated": _contact_dict(contact)}

    @tool(parse_docstring=True)
    def create_task(contact_email: str, description: str, due_in_days: int = 3) -> dict:
        """Create a follow-up task for a salesperson, attached to an existing contact.

        Args:
            contact_email: Email of the contact the task is about.
            description: What needs to be done, e.g. 'Book a demo call'.
            due_in_days: Days from today until the task is due.
            contact_email: Email of the contact the task is about.
            description: What needs to be done, e.g. 'Book a demo call'.
            due_in_days: Days from today until the task is due.
        """
        email = contact_email.strip().lower()
        with Session(engine) as s:
            contact = s.scalar(select(Contact).where(Contact.email == email))
            if contact is None:
                return {"error": f"No contact with email {email}. Create the contact first."}
            task = Task(
                contact=contact,
                description=description,
                due_date=date.today() + timedelta(days=due_in_days),
            )
            s.add(task)
            s.commit()
            return {
                "created_task": {
                    "id": task.id,
                    "contact": contact.email,
                    "description": task.description,
                    "due_date": task.due_date.isoformat(),
                }
            }

    return [search_contacts, lookup_company, create_contact, update_contact, create_task]
    



