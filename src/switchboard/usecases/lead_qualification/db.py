from datetime import UTC, date, datetime

from sqlalchemy import ForeignKey, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    domain: Mapped[str] = mapped_column(unique=True)
    industry: Mapped[str | None]
    employee_count: Mapped[int | None]

    contacts: Mapped[list["Contact"]] = relationship(back_populates="company")
    deals: Mapped[list["Deal"]] = relationship(back_populates="company")


class Contact(Base):
    __tablename__ = "contacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)
    title: Mapped[str | None]
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"))
    status: Mapped[str] = mapped_column(default="new")  # new | qualified | disqualified
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    company: Mapped[Company | None] = relationship(back_populates="contacts")
    tasks: Mapped[list["Task"]] = relationship(back_populates="contact")


class Deal(Base):
    __tablename__ = "deals"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    name: Mapped[str]
    amount: Mapped[int]  # whole euros; never store money as float
    stage: Mapped[str] = mapped_column(default="open")  # open | won | lost

    company: Mapped[Company] = relationship(back_populates="deals")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    contact_id: Mapped[int] = mapped_column(ForeignKey("contacts.id"))
    description: Mapped[str]
    due_date: Mapped[date | None]
    done: Mapped[bool] = mapped_column(default=False)

    contact: Mapped[Contact] = relationship(back_populates="tasks")


def make_engine(url: str = "sqlite:///crm.db") -> Engine:
    """Create an engine and make sure all tables exist."""
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine


def seed(session: Session) -> None:
    """Insert fake demo data. Does nothing if the CRM already has companies."""
    if session.scalar(select(Company.id).limit(1)) is not None:
        return

    acme = Company(
        name="Acme Robotics",
        domain="acmerobotics.example",
        industry="Manufacturing",
        employee_count=450,
    )
    globex = Company(
        name="Globex Logistics",
        domain="globex.example",
        industry="Logistics",
        employee_count=1200,
    )
    tinyco = Company(
        name="TinyCo Studio",
        domain="tinyco.example",
        industry="Design",
        employee_count=4,
    )

    priya = Contact(
        name="Priya Sharma",
        email="priya@acmerobotics.example",
        title="CTO",
        company=acme,
        status="qualified",
    )
    contacts = [
        priya,
        Contact(name="Jonas Weber", email="jonas@acmerobotics.example", title="IT Manager", company=acme),
        Contact(name="Mei Lin", email="mei.lin@globex.example", title="Head of Operations", company=globex),
        Contact(name="Sam Ortiz", email="sam@tinyco.example", title="Founder", company=tinyco, status="disqualified"),
    ]

    deals = [
        Deal(name="Acme - 50 seat pilot", amount=25_000, stage="open", company=acme),
        Deal(name="Globex - annual licence", amount=90_000, stage="won", company=globex),
    ]

    tasks = [
        Task(description="Send Acme pilot proposal", contact=priya, due_date=date(2026, 11, 15)),
    ]

    session.add_all([acme, globex, tinyco, *contacts, *deals, *tasks])
    session.commit()


if __name__ == "__main__":
    engine = make_engine()
    with Session(engine) as session:
        seed(session)
        for c in session.scalars(select(Contact)):
            print(f"{c.name:<14} {c.email:<30} {c.company.name if c.company else '-'}")
