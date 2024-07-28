import sys
import yaml
import joblib
import torch

sys.path.append("./src/")


class CustomException(Exception):
    def __init__(self, message: str):
        super(CustomException, self).__init__()
        self.message = message


def dump(value: str, filename: str):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        raise CustomException("Cannot be dump into pickle file".capitalize())


def load(filename: str):
    if filename is not None:
        joblib.load(filename=filename)

    else:
        raise CustomException("Cannot be load the pickle file".capitalize())


def device_init(self, device: str = "mps"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    else:
        return torch.device("cpu")


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)


english = [
    "The sun is shining brightly today",
    "I enjoy reading books on rainy afternoons",
    "The cat sat on the windowsill watching the birds",
    "She baked a delicious chocolate cake for dessert",
    "We went for a long walk in the park yesterday",
    "He plays the guitar beautifully during the evenings",
    "They traveled to Japan for their vacation last year",
    "Her garden is full of colorful flowers and butterflies",
    "The children are playing soccer in the backyard",
    "He finished his homework before dinner",
    "She loves painting landscapes and portraits",
    "The train arrived at the station on time",
    "We watched a fascinating documentary about space",
    "He enjoys hiking in the mountains on weekends",
    "The dog barked loudly at the mailman",
    "She wrote a heartfelt letter to her friend",
    "They visited the new museum downtown",
    "The coffee shop on the corner serves the best lattes",
    "He bought a new bicycle for his birthday",
    "The chef prepared a gourmet meal for the guests",
    "She practices yoga every morning to stay fit",
    "They attended a concert at the open-air theater",
    "The tree in the backyard is over a hundred years old",
    "He reads the newspaper every day at breakfast",
    "The artist displayed her work at the local gallery",
    "They went camping by the lake last weekend",
    "The librarian recommended a great mystery novel",
    "She enjoys knitting scarves and hats in her free time",
    "The bird sang a beautiful song from its perch",
    "He fixed the leaky faucet in the kitchen",
    "They organized a community clean-up event",
    "The puppy chewed on his favorite toy",
    "She planted tomatoes and herbs in her garden",
    "The car broke down on the highway",
    "He attended a workshop on digital marketing",
    "The sun set over the horizon, painting the sky orange",
    "She found a rare coin while digging in the garden",
    "They danced under the stars at the outdoor party",
    "The cat purred contentedly on her lap",
    "He studied for his exams late into the night",
    "The fireworks lit up the night sky during the festival",
    "She baked homemade cookies for her neighbors",
    "They went on a road trip along the coast",
    "The child built a sandcastle on the beach",
    "He wrote a novel during his summer vacation",
    "The team celebrated their victory after the match",
    "She attended a seminar on climate change",
    "The kite flew high above the field",
    "He enjoys fishing at the river early in the morning",
    "The flowers in the garden bloomed beautifully",
    "She volunteered at the animal shelter on weekends",
    "They had a barbecue in the backyard with friends",
    "The squirrel gathered acorns for the winter",
    "He took a photograph of the stunning sunset",
    "The movie was a thrilling adventure from start to finish",
    "She taught her dog a new trick",
    "They explored the ancient ruins during their trip",
    "The bakery on Main Street has the best pastries",
    "He played chess with his grandfather",
    "The rain created puddles on the sidewalk",
    "She sewed a dress for the upcoming party",
    "They planted trees as part of the reforestation project",
    "The butterfly landed gently on the flower",
    "He jogged around the park every morning",
    "The book club met to discuss the latest novel",
    "She painted the walls of her room a bright yellow",
    "They watched the stars through a telescope",
    "The ice cream truck drove through the neighborhood",
    "He built a birdhouse and hung it in the garden",
    "The musician composed a new song",
    "She attended a pottery class on Saturday",
    "They enjoyed a picnic by the lake",
    "The wind blew softly through the trees",
    "He solved the puzzle after hours of thinking",
    "The library held a book sale to raise funds",
    "She decorated the living room with fresh flowers",
    "They traveled to the mountains for a ski trip",
    "The dog wagged its tail happily",
    "He learned to cook a new recipe",
    "The rain fell steadily throughout the night",
    "She crafted a handmade gift for her friend",
    "They celebrated their anniversary with a special dinner",
    "The kitten chased a ball of yarn around the room",
    "He repaired the fence in the backyard",
    "The ocean waves crashed against the shore",
    "She played the piano beautifully",
    "They attended a workshop on photography",
    "The leaves changed color in the fall",
    "He wrote a poem about the beauty of nature",
    "The children played hide and seek in the garden",
    "She baked a loaf of bread from scratch",
    "They went on a boat ride across the lake",
    "The campfire crackled and popped",
    "He collected stamps from different countries",
    "The rain washed away the dust from the streets",
    "She drew a portrait of her best friend",
    "They celebrated the holiday with family and friends",
    "The owl hooted softly in the night",
    "He planted a rose bush in the garden",
    "The sunflowers turned towards the sun",
    "She practiced her speech for the conference",
    "They went snorkeling in the coral reef",
]

german = [
    "Die Sonne scheint heute hell",
    "Ich lese gerne Bücher an regnerischen Nachmittagen",
    "Die Katze saß auf der Fensterbank und beobachtete die Vögel",
    "Sie hat einen leckeren Schokoladenkuchen zum Nachtisch gebacken",
    "Wir sind gestern lange im Park spazieren gegangen",
    "Er spielt abends wunderschön Gitarre",
    "Sie sind letztes Jahr nach Japan in den Urlaub gefahren",
    "Ihr Garten ist voller bunter Blumen und Schmetterlinge",
    "Die Kinder spielen im Hinterhof Fußball",
    "Er hat seine Hausaufgaben vor dem Abendessen fertig gemacht",
    "Sie malt gerne Landschaften und Porträts",
    "Der Zug kam pünktlich am Bahnhof an",
    "Wir haben eine faszinierende Dokumentation über den Weltraum gesehen",
    "Er wandert gerne am Wochenende in den Bergen",
    "Der Hund hat den Briefträger laut angebellt",
    "Sie hat einen herzlichen Brief an ihre Freundin geschrieben",
    "Sie haben das neue Museum in der Innenstadt besucht",
    "Das Café an der Ecke serviert die besten Lattes",
    "Er hat sich zum Geburtstag ein neues Fahrrad gekauft",
    "Der Koch hat für die Gäste ein Gourmetessen zubereitet",
    "Sie macht jeden Morgen Yoga, um fit zu bleiben",
    "Sie haben ein Konzert im Freilufttheater besucht",
    "Der Baum im Hinterhof ist über hundert Jahre alt",
    "Er liest jeden Tag beim Frühstück die Zeitung",
    "Die Künstlerin hat ihre Werke in der lokalen Galerie ausgestellt",
    "Sie sind letztes Wochenende am See campen gegangen",
    "Der Bibliothekar hat einen tollen Kriminalroman empfohlen",
    "Sie strickt gerne Schals und Mützen in ihrer Freizeit",
    "Der Vogel sang ein wunderschönes Lied von seiner Stange",
    "Er hat den tropfenden Wasserhahn in der Küche repariert",
    "Sie haben eine Gemeinschaftsreinigungsaktion organisiert",
    "Der Welpe kaute an seinem Lieblingsspielzeug",
    "Sie hat Tomaten und Kräuter in ihrem Garten gepflanzt",
    "Das Auto ist auf der Autobahn liegen geblieben",
    "Er hat einen Workshop über digitales Marketing besucht",
    "Die Sonne ging über dem Horizont unter und malte den Himmel orange",
    "Sie hat beim Graben im Garten eine seltene Münze gefunden",
    "Sie haben bei der Outdoor-Party unter den Sternen getanzt",
    "Die Katze schnurrte zufrieden auf ihrem Schoß",
    "Er hat bis spät in die Nacht für seine Prüfungen gelernt",
    "Die Feuerwerke erleuchteten den Nachthimmel während des Festivals",
    "Sie hat selbstgebackene Kekse für ihre Nachbarn gemacht",
    "Sie sind entlang der Küste auf einen Roadtrip gegangen",
    "Das Kind baute eine Sandburg am Strand",
    "Er hat während seines Sommerurlaubs einen Roman geschrieben",
    "Das Team feierte ihren Sieg nach dem Spiel",
    "Sie hat ein Seminar über den Klimawandel besucht",
    "Der Drachen flog hoch über dem Feld",
    "Er geht gerne früh morgens am Fluss angeln",
    "Die Blumen im Garten blühten wunderschön",
    "Sie hat am Wochenende im Tierheim freiwillig gearbeitet",
    "Sie haben eine Grillparty im Hinterhof mit Freunden gemacht",
    "Das Eichhörnchen sammelte Eicheln für den Winter",
    "Er hat ein Foto von dem atemberaubenden Sonnenuntergang gemacht",
    "Der Film war ein spannendes Abenteuer von Anfang bis Ende",
    "Sie hat ihrem Hund einen neuen Trick beigebracht",
    "Sie haben während ihrer Reise die antiken Ruinen erkundet",
    "Die Bäckerei in der Hauptstraße hat die besten Gebäckstücke",
    "Er hat Schach mit seinem Großvater gespielt",
    "Der Regen hat Pfützen auf dem Bürgersteig gebildet",
    "Sie hat ein Kleid für die bevorstehende Party genäht",
    "Sie haben als Teil des Wiederaufforstungsprojekts Bäume gepflanzt",
    "Der Schmetterling landete sanft auf der Blume",
    "Er joggt jeden Morgen um den Park",
    "Der Buchclub traf sich, um den neuesten Roman zu diskutieren",
    "Sie hat die Wände ihres Zimmers in einem hellen Gelb gestrichen",
    "Sie haben die Sterne durch ein Teleskop beobachtet",
    "Der Eiswagen fuhr durch die Nachbarschaft",
    "Er hat ein Vogelhaus gebaut und es im Garten aufgehängt",
    "Der Musiker hat ein neues Lied komponiert",
    "Sie hat am Samstag an einem Töpferkurs teilgenommen",
    "Sie haben ein Picknick am See genossen",
    "Der Wind wehte sanft durch die Bäume",
    "Er hat das Puzzle nach stundenlangem Nachdenken gelöst",
    "Die Bibliothek hat einen Bücherverkauf veranstaltet, um Spenden zu sammeln",
    "Sie hat das Wohnzimmer mit frischen Blumen dekoriert",
    "Sie sind für einen Skiausflug in die Berge gefahren",
    "Der Hund wedelte glücklich mit dem Schwanz",
    "Er hat ein neues Rezept gelernt zu kochen",
    "Der Regen fiel die ganze Nacht hindurch gleichmäßig",
    "Sie hat ein handgemachtes Geschenk für ihre Freundin gebastelt",
    "Sie haben ihr Jubiläum mit einem besonderen Abendessen gefeiert",
    "Das Kätzchen jagte ein Wollknäuel im Zimmer herum",
    "Er hat den Zaun im Hinterhof repariert",
    "Die Meereswellen krachten gegen die Küste",
    "Sie hat wunderschön Klavier gespielt",
    "Sie haben einen Workshop über Fotografie besucht",
    "Die Blätter wechselten im Herbst die Farbe",
    "Er hat ein Gedicht über die Schönheit der Natur geschrieben",
    "Die Kinder spielten Verstecken im Garten",
    "Sie hat ein Brot von Grund auf gebacken",
    "Sie sind mit dem Boot über den See gefahren",
    "Das Lagerfeuer knisterte und knallte",
    "Er sammelte Briefmarken aus verschiedenen Ländern",
    "Der Regen hat den Staub von den Straßen gewaschen",
    "Sie hat ein Porträt ihrer besten Freundin gezeichnet",
    "Sie haben die Feiertage mit Familie und Freunden gefeiert",
    "Die Eule hat in der Nacht leise geheult",
    "Er hat einen Rosenbusch im Garten gepflanzt",
    "Die Sonnenblumen drehten sich zur Sonne",
    "Sie hat ihre Rede für die Konferenz geübt",
    "Sie sind im Korallenriff schnorcheln gegangen",
]
