
import os
import platform
import subprocess
import json

category_map = {
# These created errors when mapping categories to descriptions
'acc-phys': 'Accelerator Physics',
'adap-org': 'Not available',
'q-bio': 'Not available',
'cond-mat': 'Not available',
'chao-dyn': 'Not available',
'patt-sol': 'Not available',
'dg-ga': 'Not available',
'solv-int': 'Not available',
'bayes-an': 'Not available',
'comp-gas': 'Not available',
'alg-geom': 'Not available',
'funct-an': 'Not available',
'q-alg': 'Not available',
'ao-sci': 'Not available',
'atom-ph': 'Atomic Physics',
'chem-ph': 'Chemical Physics',
'plasm-ph': 'Plasma Physics',
'mtrl-th': 'Not available',
'cmp-lg': 'Not available',
'supr-con': 'Not available',
###

# Added
'econ.GN': 'General Economics',
'econ.TH': 'Theoretical Economics',
'eess.SY': 'Systems and Control',

'astro-ph': 'Astrophysics',
'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
'astro-ph.EP': 'Earth and Planetary Astrophysics',
'astro-ph.GA': 'Astrophysics of Galaxies',
'astro-ph.HE': 'High Energy Astrophysical Phenomena',
'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
'astro-ph.SR': 'Solar and Stellar Astrophysics',
'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
'cond-mat.mtrl-sci': 'Materials Science',
'cond-mat.other': 'Other Condensed Matter',
'cond-mat.quant-gas': 'Quantum Gases',
'cond-mat.soft': 'Soft Condensed Matter',
'cond-mat.stat-mech': 'Statistical Mechanics',
'cond-mat.str-el': 'Strongly Correlated Electrons',
'cond-mat.supr-con': 'Superconductivity',
'cs.AI': 'Artificial Intelligence',
'cs.AR': 'Hardware Architecture',
'cs.CC': 'Computational Complexity',
'cs.CE': 'Computational Engineering, Finance, and Science',
'cs.CG': 'Computational Geometry',
'cs.CL': 'Computation and Language',
'cs.CR': 'Cryptography and Security',
'cs.CV': 'Computer Vision and Pattern Recognition',
'cs.CY': 'Computers and Society',
'cs.DB': 'Databases',
'cs.DC': 'Distributed, Parallel, and Cluster Computing',
'cs.DL': 'Digital Libraries',
'cs.DM': 'Discrete Mathematics',
'cs.DS': 'Data Structures and Algorithms',
'cs.ET': 'Emerging Technologies',
'cs.FL': 'Formal Languages and Automata Theory',
'cs.GL': 'General Literature',
'cs.GR': 'Graphics',
'cs.GT': 'Computer Science and Game Theory',
'cs.HC': 'Human-Computer Interaction',
'cs.IR': 'Information Retrieval',
'cs.IT': 'Information Theory',
'cs.LG': 'Machine Learning',
'cs.LO': 'Logic in Computer Science',
'cs.MA': 'Multiagent Systems',
'cs.MM': 'Multimedia',
'cs.MS': 'Mathematical Software',
'cs.NA': 'Numerical Analysis',
'cs.NE': 'Neural and Evolutionary Computing',
'cs.NI': 'Networking and Internet Architecture',
'cs.OH': 'Other Computer Science',
'cs.OS': 'Operating Systems',
'cs.PF': 'Performance',
'cs.PL': 'Programming Languages',
'cs.RO': 'Robotics',
'cs.SC': 'Symbolic Computation',
'cs.SD': 'Sound',
'cs.SE': 'Software Engineering',
'cs.SI': 'Social and Information Networks',
'cs.SY': 'Systems and Control',
'econ.EM': 'Econometrics',
'eess.AS': 'Audio and Speech Processing',
'eess.IV': 'Image and Video Processing',
'eess.SP': 'Signal Processing',
'gr-qc': 'General Relativity and Quantum Cosmology',
'hep-ex': 'High Energy Physics - Experiment',
'hep-lat': 'High Energy Physics - Lattice',
'hep-ph': 'High Energy Physics - Phenomenology',
'hep-th': 'High Energy Physics - Theory',
'math.AC': 'Commutative Algebra',
'math.AG': 'Algebraic Geometry',
'math.AP': 'Analysis of PDEs',
'math.AT': 'Algebraic Topology',
'math.CA': 'Classical Analysis and ODEs',
'math.CO': 'Combinatorics',
'math.CT': 'Category Theory',
'math.CV': 'Complex Variables',
'math.DG': 'Differential Geometry',
'math.DS': 'Dynamical Systems',
'math.FA': 'Functional Analysis',
'math.GM': 'General Mathematics',
'math.GN': 'General Topology',
'math.GR': 'Group Theory',
'math.GT': 'Geometric Topology',
'math.HO': 'History and Overview',
'math.IT': 'Information Theory',
'math.KT': 'K-Theory and Homology',
'math.LO': 'Logic',
'math.MG': 'Metric Geometry',
'math.MP': 'Mathematical Physics',
'math.NA': 'Numerical Analysis',
'math.NT': 'Number Theory',
'math.OA': 'Operator Algebras',
'math.OC': 'Optimization and Control',
'math.PR': 'Probability',
'math.QA': 'Quantum Algebra',
'math.RA': 'Rings and Algebras',
'math.RT': 'Representation Theory',
'math.SG': 'Symplectic Geometry',
'math.SP': 'Spectral Theory',
'math.ST': 'Statistics Theory',
'math-ph': 'Mathematical Physics',
'nlin.AO': 'Adaptation and Self-Organizing Systems',
'nlin.CD': 'Chaotic Dynamics',
'nlin.CG': 'Cellular Automata and Lattice Gases',
'nlin.PS': 'Pattern Formation and Solitons',
'nlin.SI': 'Exactly Solvable and Integrable Systems',
'nucl-ex': 'Nuclear Experiment',
'nucl-th': 'Nuclear Theory',
'physics.acc-ph': 'Accelerator Physics',
'physics.ao-ph': 'Atmospheric and Oceanic Physics',
'physics.app-ph': 'Applied Physics',
'physics.atm-clus': 'Atomic and Molecular Clusters',
'physics.atom-ph': 'Atomic Physics',
'physics.bio-ph': 'Biological Physics',
'physics.chem-ph': 'Chemical Physics',
'physics.class-ph': 'Classical Physics',
'physics.comp-ph': 'Computational Physics',
'physics.data-an': 'Data Analysis, Statistics and Probability',
'physics.ed-ph': 'Physics Education',
'physics.flu-dyn': 'Fluid Dynamics',
'physics.gen-ph': 'General Physics',
'physics.geo-ph': 'Geophysics',
'physics.hist-ph': 'History and Philosophy of Physics',
'physics.ins-det': 'Instrumentation and Detectors',
'physics.med-ph': 'Medical Physics',
'physics.optics': 'Optics',
'physics.plasm-ph': 'Plasma Physics',
'physics.pop-ph': 'Popular Physics',
'physics.soc-ph': 'Physics and Society',
'physics.space-ph': 'Space Physics',
'q-bio.BM': 'Biomolecules',
'q-bio.CB': 'Cell Behavior',
'q-bio.GN': 'Genomics',
'q-bio.MN': 'Molecular Networks',
'q-bio.NC': 'Neurons and Cognition',
'q-bio.OT': 'Other Quantitative Biology',
'q-bio.PE': 'Populations and Evolution',
'q-bio.QM': 'Quantitative Methods',
'q-bio.SC': 'Subcellular Processes',
'q-bio.TO': 'Tissues and Organs',
'q-fin.CP': 'Computational Finance',
'q-fin.EC': 'Economics',
'q-fin.GN': 'General Finance',
'q-fin.MF': 'Mathematical Finance',
'q-fin.PM': 'Portfolio Management',
'q-fin.PR': 'Pricing of Securities',
'q-fin.RM': 'Risk Management',
'q-fin.ST': 'Statistical Finance',
'q-fin.TR': 'Trading and Market Microstructure',
'quant-ph': 'Quantum Physics',
'stat.AP': 'Applications',
'stat.CO': 'Computation',
'stat.ME': 'Methodology',
'stat.ML': 'Machine Learning',
'stat.OT': 'Other Statistics',
'stat.TH': 'Statistics Theory'
}

def check_library(library_name):
    try:
        __import__(library_name)
        print(f"✅ {library_name} is installed.")
        # Optionally, get the version
        module = __import__(library_name)
        version = getattr(module, "__version__", "Version not available")
        print(f"Version: {version}")
        return True
    except ImportError:
        print(f"❌ {library_name} is not installed.")
        print("Please install using pip or conda")
        exit()

def check_system():
    system_name = platform.system()
    return system_name

def download_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    download_path = os.path.join(current_dir, "data")
    if os.path.exists(os.path.join(current_dir, download_path, "arxiv-metadata-oai-snapshot.json")):
        print("Data already exist, skip download.")
        return 
    
    system_name = check_system()
    # Define the path to `kaggle.json` in the same directory as the script
    kaggle_json_path = os.path.join(current_dir, "kaggle.json")
    # Ensure the Kaggle JSON file exists
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(f"kaggle.json not found in {current_dir}. Please place it there.")

    # Set the environment variables for Kaggle API authentication
    with open(kaggle_json_path, "r") as f:
        credentials = json.load(f)
        os.environ["KAGGLE_USERNAME"] = credentials["username"]
        os.environ["KAGGLE_KEY"] = credentials["key"]

    if system_name == "Windows":
        kaggle_path = os.path.expanduser(os.path.join(current_dir, "./data"))
        os.makedirs(kaggle_path, exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "Cornell-University/arxiv",
            "--unzip",
            "-p", download_path  # Set download path
        ], check=True)

    elif system_name == "Darwin":  # macOS
        kaggle_path = os.path.expanduser(os.path.join(current_dir, "./data"))
        os.makedirs(kaggle_path, exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "Cornell-University/arxiv",
            "--unzip",
            "-p", download_path  # Set download path
        ], check=True)

    elif system_name == "Linux":
        kaggle_path = os.path.expanduser(os.path.join(current_dir, "./data"))
        os.makedirs(kaggle_path, exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "Cornell-University/arxiv",
            "--unzip",
            "-p", download_path  # Set download path
        ], check=True)

    else:
        print("Unsupported OS:", system_name)

    print(f"✅ Dataset downloaded to: {download_path}")

def get_cat_text(x):

    cat_text = ''

    # Put the codes into a list
    cat_list = x.split(' ')

    for i, item in enumerate(cat_list):

        cat_name = category_map[item]

        # If there was no description available
        # for the category code then don't include it in the text.
        if cat_name != 'Not available':

            if i == 0:
                cat_text = cat_name
            else:
                cat_text = cat_text + ', ' + cat_name

    # Remove leading and trailing spaces
    cat_text = cat_text.strip()

    return cat_text

def clean_text(x):

    # Replace newline characters with a space
    new_text = x.replace("\n", " ")
    # Remove leading and trailing spaces
    new_text = new_text.strip()

    return new_text

# download_data()