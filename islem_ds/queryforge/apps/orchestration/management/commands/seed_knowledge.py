from django.core.management.base import BaseCommand
from apps.orchestration.models import KnowledgeItem

class Command(BaseCommand):
    help = 'Seeds the database with sample knowledge items'

    def handle(self, *args, **kwargs):
        items = [
            {"title": "Python Dictionary", "content": "Dictionaries are used to store data values in key:value pairs. A dictionary is a collection which is ordered, changeable and do not allow duplicates.", "source": "Python Docs"},
            {"title": "Django ORM", "content": "The Django object-relational mapping (ORM) lets you interact with a database using Python, rather than SQL.", "source": "Django Docs"},
            {"title": "Gravity", "content": "Gravity is a fundamental interaction which causes mutual attraction between all things that have mass.", "source": "Science textbook"},
            {"title": "Photosynthesis", "content": "Photosynthesis is a process used by plants to convert light energy into chemical energy.", "source": "Biology Book"},
            {"title": "Water Boiling Point", "content": "The boiling point of water is 100 degrees Celsius or 212 degrees Fahrenheit at sea level.", "source": "Science Fact"},
            {"title": "Earth Circumference", "content": "The circumference of the Earth at the equator is about 40,075 kilometers (24,901 miles).", "source": "Geography Fact"},
            {"title": "Mount Everest", "content": "Mount Everest is Earth's highest mountain above sea level, reaching 8,848 meters (29,029 ft).", "source": "Geography Fact"},
            {"title": "Speed of Light", "content": "The speed of light in a vacuum is exactly 299,792,458 meters per second.", "source": "Physics Fact"},
            {"title": "Avogadro's Number", "content": "Avogadro's number is 6.022 x 10^23, representing the number of particles in a mole.", "source": "Chemistry Fact"},
            {"title": "Django Views", "content": "A view function, or view for short, is a Python function that takes a Web request and returns a Web response.", "source": "Django Docs"},
            {"title": "Python Decorators", "content": "A decorator is a design pattern in Python that allows a user to add new functionality to an existing object without modifying its structure.", "source": "Python Docs"},
            {"title": "Jinja2 Templates", "content": "Jinja2 is a modern and designer-friendly templating language for Python, modelled after Django's templates.", "source": "Jinja Docs"},
            {"title": "List Comprehensions", "content": "List comprehensions provide a concise way to create lists in Python.", "source": "Python Docs"},
            {"title": "Capital of France", "content": "Paris is the capital and most populous city of France.", "source": "Geography Fact"},
            {"title": "Mars", "content": "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System.", "source": "Astronomy Fact"},
            {"title": "E=mc^2", "content": "E=mc^2 is the mass-energy equivalence equation proposed by Albert Einstein.", "source": "Physics Fact"},
            {"title": "Mona Lisa", "content": "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.", "source": "Art Fact"},
            {"title": "Shakespeare", "content": "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language.", "source": "History Fact"},
            {"title": "World War II", "content": "World War II or the Second World War was a global conflict that lasted from 1939 to 1945.", "source": "History Fact"},
            {"title": "SQL", "content": "SQL (Structured Query Language) is a standard language for storing, manipulating and retrieving data in databases.", "source": "Tech Standard"}
        ]

        count = 0
        for item in items:
            obj, created = KnowledgeItem.objects.get_or_create(
                title=item["title"],
                defaults={
                    "content": item["content"],
                    "source": item["source"]
                }
            )
            if created:
                count += 1

        self.stdout.write(self.style.SUCCESS(f'Successfully seeded {count} knowledge items.'))
