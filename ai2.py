
# ==============================================================
# RESUME SCREENER AND JOB MATCHER
# AI PBL PROJECT
#
# AI Techniques:
# 1. Rule-Based Expert System
# 2. Breadth First Search (BFS)
# 3. Depth First Search (DFS)
# 4. TF-IDF
# 5. Best-First Search
#
# GUI: Tkinter
# PDF Processing: pypdf
# ==============================================================

import re
import math
import heapq
import importlib
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# Load pypdf when available, with PyPDF2 as a compatibility fallback.
# importlib avoids a missing-module warning before the dependency is installed.
try:
    PdfReader = importlib.import_module("pypdf").PdfReader
except ImportError:
    try:
        PdfReader = importlib.import_module("PyPDF2").PdfReader
    except ImportError:
        PdfReader = None


# ==============================================================
# 1. KNOWLEDGE BASE
# ==============================================================

ontology = {
    "tech": ["programming", "data", "web"],
    "programming": ["python", "java", "c++"],
    "data": ["sql", "machine learning", "statistics", "excel"],
    "web": ["html", "css", "javascript", "django"],
}

related = {
    "python": ["machine learning", "django", "sql"],
    "machine learning": ["python", "statistics"],
    "statistics": ["machine learning", "excel"],
    "excel": ["statistics", "sql"],
    "sql": ["python", "excel"],
    "django": ["python", "html"],
    "html": ["css", "javascript", "django"],
    "css": ["html"],
    "javascript": ["html"],
    "java": ["c++"],
    "c++": ["java"],
}

synonyms = {
    "ml": "machine learning",
    "js": "javascript",
    "cpp": "c++",
    "stats": "statistics",
    "ms excel": "excel",
}

all_skills = set(related.keys())


# ==============================================================
# 2. JOB DATABASE
# ==============================================================

jobs = [
    {
        "title": "Data Analyst",
        "required": ["sql", "excel", "statistics"],
        "min_cgpa": 6.5,
        "min_exp": 0,
        "degrees": ["btech", "bsc", "bca", "mca"],
        "desc": "analyse data using sql and excel, make reports, use statistics to find trends",
    },
    {
        "title": "ML Engineer",
        "required": ["python", "machine learning", "statistics"],
        "min_cgpa": 7.5,
        "min_exp": 1,
        "degrees": ["btech", "mtech"],
        "desc": "build and train machine learning models in python, work with data and statistics",
    },
    {
        "title": "Web Developer",
        "required": ["html", "css", "javascript"],
        "min_cgpa": 6.0,
        "min_exp": 0,
        "degrees": ["btech", "bca", "bsc"],
        "desc": "make websites using html css and javascript, build responsive pages",
    },
    {
        "title": "Backend Developer",
        "required": ["python", "django", "sql"],
        "min_cgpa": 7.0,
        "min_exp": 1,
        "degrees": ["btech", "mca"],
        "desc": "build server side apps with python and django, design databases using sql",
    },
]


# ==============================================================
# 3. SAMPLE RESUMES
# ==============================================================

sample_resumes = {
    "Riya Sharma": """Name: Riya Sharma
Degree: BTech
CGPA: 8.4
Experience: 1 year
Skills: python, ml, statistics, sql
Projects: Built a house price predictor using machine learning and python.""",

    "Aman Verma": """Name: Aman Verma
Degree: BCA
CGPA: 6.8
Experience: 0 years
Skills: html, css, js
Projects: Made a portfolio website and a to-do app using javascript.""",

    "Neha Gupta": """Name: Neha Gupta
Degree: BSc
CGPA: 5.9
Experience: 2 years
Skills: excel, sql
Projects: Sales dashboard in excel for a local shop."""
}


# ==============================================================
# 4. DFS ALGORITHM
# ==============================================================

def dfs_skills(node, visited=None):

    if visited is None:
        visited = []

    if node in visited:
        return visited

    visited.append(node)

    for child in ontology.get(node, []):
        dfs_skills(child, visited)

    return visited


# ==============================================================
# 5. BFS ALGORITHM
# ==============================================================

def bfs_distance(start, goal):

    if start == goal:
        return 0

    queue = deque([(start, 0)])
    seen = {start}

    while queue:

        skill, dist = queue.popleft()

        for nxt in related.get(skill, []):

            if nxt == goal:
                return dist + 1

            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))

    return -1


# ==============================================================
# 6. RESUME SCANNER
# ==============================================================

def clean(text):
    return re.sub(r"[^a-z0-9+\s]", " ", text.lower())


def scan_resume(text):

    info = {
        "name": "Unknown",
        "degree": "",
        "cgpa": 0.0,
        "exp": 0,
        "skills": set()
    }

    m = re.search(
        r"name\s*:\s*(.*?)(?=\s+(?:degree|cgpa|experience|skills|projects)\s*:|$)",
        text, re.I
    )

    if m:
        info["name"] = m.group(1).strip()

    m = re.search(r"degree\s*:\s*(.+)", text, re.I)

    if m:
        info["degree"] = re.sub(
            r"[^a-z]", "",
            m.group(1).lower().split()[0]
        )

    m = re.search(r"cgpa\s*:\s*([\d.]+)", text, re.I)

    if m:
        info["cgpa"] = float(m.group(1))

    m = re.search(r"experience\s*:\s*(\d+)", text, re.I)

    if m:
        info["exp"] = int(m.group(1))

    words = clean(text)

    for short, full in synonyms.items():
        words = re.sub(
            r"\b" + re.escape(short) + r"\b",
            full,
            words
        )

    for skill in all_skills:

        if re.search(
            r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])",
            words
        ):
            info["skills"].add(skill)

    return info


# ==============================================================
# 7. TF-IDF ALGORITHM
# ==============================================================

def tokenize(text):
    return clean(text).split()


def tf_idf_similarity(resume_text, job_desc, all_docs):

    docs = [tokenize(d) for d in all_docs]
    n = len(docs)

    def idf(word):

        count = sum(1 for d in docs if word in d)

        return math.log((n + 1) / (count + 1)) + 1

    def vector(tokens):

        vec = {}

        if not tokens:
            return vec

        for w in set(tokens):

            tf = tokens.count(w) / len(tokens)

            vec[w] = tf * idf(w)

        return vec

    v1 = vector(tokenize(resume_text))
    v2 = vector(tokenize(job_desc))

    dot = sum(
        v1[w] * v2[w]
        for w in v1 if w in v2
    )

    mag1 = math.sqrt(sum(x * x for x in v1.values()))
    mag2 = math.sqrt(sum(x * x for x in v2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)


# ==============================================================
# 8. RULE-BASED EXPERT SYSTEM
# ==============================================================

def rule_cgpa(info, job):

    ok = info["cgpa"] >= job["min_cgpa"]

    return ok, f"CGPA: {info['cgpa']} (Required: {job['min_cgpa']})"


def rule_experience(info, job):

    ok = info["exp"] >= job["min_exp"]

    return ok, f"Experience: {info['exp']} years (Required: {job['min_exp']})"


def rule_degree(info, job):

    ok = info["degree"] in job["degrees"]

    return ok, (
        f"Degree: {info['degree'] or 'Not provided'} "
        f"(Allowed: {', '.join(job['degrees'])})"
    )


rules = [rule_cgpa, rule_experience, rule_degree]


def check_eligibility(info, job):

    reasons = []
    eligible = True

    for rule in rules:

        passed, msg = rule(info, job)

        reasons.append(
            ("PASS - " if passed else "FAIL - ") + msg
        )

        if not passed:
            eligible = False

    return eligible, reasons


# ==============================================================
# 9. JOB MATCHING AND BEST-FIRST SEARCH
# ==============================================================

def skill_score(info, job):

    total = 0
    missing = []

    for req in job["required"]:

        if req in info["skills"]:
            total += 1

        else:

            close = False

            for have in info["skills"]:

                if bfs_distance(have, req) == 1:
                    close = True
                    break

            if close:
                total += 0.5

            missing.append(req)

    return total / len(job["required"]), missing


def match_jobs(resume_text):

    info = scan_resume(resume_text)

    all_docs = [resume_text] + [
        j["desc"] for j in jobs
    ]

    heap = []
    results = {}

    for i, job in enumerate(jobs):

        s_score, missing = skill_score(info, job)

        t_score = tf_idf_similarity(
            resume_text,
            job["desc"],
            all_docs
        )

        eligible, reasons = check_eligibility(info, job)

        final = 0.6 * s_score + 0.4 * t_score

        if not eligible:
            final *= 0.5

        results[i] = (
            final,
            s_score,
            t_score,
            eligible,
            reasons,
            missing
        )

        heapq.heappush(heap, (-final, i))

    ranked = []

    while heap:

        neg, i = heapq.heappop(heap)

        ranked.append((i, results[i]))

    return info, ranked


# ==============================================================
# 10. PROFESSIONAL GUI
# ==============================================================

class ResumeScreenerGUI:

    def __init__(self, root):

        self.root = root

        self.root.title("TalentMatch | Resume Screener")

        self.root.geometry("1250x800")
        self.root.minsize(1000, 680)

        self.root.configure(bg="#f1f5f9")

        self.current_file = ""

        self.info = None
        self.ranked = []

        self.setup_styles()
        self.create_gui()

    # ----------------------------------------------------------
    # STYLING
    # ----------------------------------------------------------

    def setup_styles(self):

        style = ttk.Style()

        style.theme_use("clam")

        style.configure(
            "Treeview",
            background="white",
            foreground="#1e293b",
            rowheight=35,
            fieldbackground="white",
            font=("Segoe UI", 10)
        )

        style.configure(
            "Treeview.Heading",
            background="#e2e8f0",
            foreground="#334155",
            font=("Segoe UI", 10, "bold"),
            padding=10
        )

        style.map(
            "Treeview",
            background=[("selected", "#dbeafe")],
            foreground=[("selected", "#1e40af")]
        )

    # ----------------------------------------------------------
    # HELPER FOR BUTTONS
    # ----------------------------------------------------------

    def make_button(self, parent, text, command,
                    bg="#2563eb", fg="white"):

        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            padx=14,
            pady=10
        )

    # ----------------------------------------------------------
    # MAIN INTERFACE
    # ----------------------------------------------------------

    def create_gui(self):

        # Header

        header = tk.Frame(
            self.root,
            bg="#172554",
            height=85
        )

        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="TalentMatch",
            font=("Segoe UI", 23, "bold"),
            fg="white",
            bg="#172554"
        ).pack(side="left", padx=28, pady=20)

        tk.Label(
            header,
            text="AI RESUME SCREENING & JOB MATCHING",
            font=("Segoe UI", 10, "bold"),
            fg="#bfdbfe",
            bg="#172554"
        ).pack(side="right", padx=28)

        # Main container

        body = tk.Frame(
            self.root,
            bg="#f1f5f9"
        )

        body.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=18
        )

        # Sidebar

        sidebar = tk.Frame(
            body,
            bg="#1e293b",
            width=220
        )

        sidebar.pack(
            side="left",
            fill="y",
            padx=(0, 18)
        )

        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="WORKSPACE",
            font=("Segoe UI", 10, "bold"),
            fg="#94a3b8",
            bg="#1e293b"
        ).pack(anchor="w", padx=20, pady=(25, 15))

        for item in [
            "Resume Analysis",
            "Job Recommendations",
            "Skill Matching",
            "AI Techniques"
        ]:

            tk.Label(
                sidebar,
                text="   " + item,
                font=("Segoe UI", 10),
                fg="white",
                bg="#1e293b",
                anchor="w",
                pady=12
            ).pack(fill="x", padx=8)

        tk.Label(
            sidebar,
            text="AI PBL PROJECT\n\n"
                 "BFS | DFS\n"
                 "TF-IDF\n"
                 "Expert System\n"
                 "Best-First Search",
            font=("Segoe UI", 10),
            fg="#bfdbfe",
            bg="#1e293b",
            justify="left"
        ).pack(side="bottom", anchor="w", padx=20, pady=25)

        # Main content

        content = tk.Frame(
            body,
            bg="#f1f5f9"
        )

        content.pack(
            side="right",
            fill="both",
            expand=True
        )

        # Page heading

        tk.Label(
            content,
            text="Resume Dashboard",
            font=("Segoe UI", 21, "bold"),
            bg="#f1f5f9",
            fg="#0f172a"
        ).pack(anchor="w")

        tk.Label(
            content,
            text="Upload a candidate resume and discover suitable job opportunities.",
            font=("Segoe UI", 10),
            bg="#f1f5f9",
            fg="#64748b"
        ).pack(anchor="w", pady=(3, 15))

        # Upload card

        upload = tk.Frame(
            content,
            bg="white",
            padx=18,
            pady=15
        )

        upload.pack(fill="x", pady=(0, 15))

        tk.Label(
            upload,
            text="UPLOAD RESUME",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#1e293b"
        ).pack(anchor="w")

        tk.Label(
            upload,
            text="Select a PDF resume or paste resume text below.",
            font=("Segoe UI", 9),
            bg="white",
            fg="#64748b"
        ).pack(anchor="w", pady=5)

        button_row = tk.Frame(upload, bg="white")
        button_row.pack(fill="x", pady=8)

        self.make_button(
            button_row,
            "Upload PDF",
            self.upload_pdf
        ).pack(side="left", padx=(0, 8))

        self.make_button(
            button_row,
            "Load Sample",
            self.load_sample,
            "#475569"
        ).pack(side="left", padx=8)

        self.make_button(
            button_row,
            "Analyze Resume",
            self.analyze_resume,
            "#059669"
        ).pack(side="left", padx=8)

        self.make_button(
            button_row,
            "Clear",
            self.clear_all,
            "#dc2626"
        ).pack(side="right")

        self.file_label = tk.Label(
            upload,
            text="No file selected",
            font=("Segoe UI", 9),
            bg="white",
            fg="#64748b"
        )

        self.file_label.pack(anchor="w", pady=4)

        self.resume_input = ScrolledText(
            upload,
            height=6,
            font=("Segoe UI", 10),
            wrap="word",
            relief="solid",
            bd=1,
            bg="#f8fafc"
        )

        self.resume_input.pack(fill="x", pady=(8, 0))

        # Candidate information cards

        cards = tk.Frame(content, bg="#f1f5f9")
        cards.pack(fill="x", pady=(0, 15))

        self.card_values = {}

        card_data = [
            ("Candidate", "candidate", "#2563eb"),
            ("CGPA", "cgpa", "#0891b2"),
            ("Experience", "experience", "#7c3aed"),
            ("Skills Found", "skills", "#059669")
        ]

        for title, key, color in card_data:

            card = tk.Frame(
                cards,
                bg="white",
                padx=12,
                pady=12
            )

            card.pack(
                side="left",
                fill="both",
                expand=True,
                padx=(0, 10)
            )

            tk.Label(
                card,
                text=title.upper(),
                font=("Segoe UI", 8, "bold"),
                fg="#64748b",
                bg="white"
            ).pack(anchor="w")

            value = tk.Label(
                card,
                text="--",
                font=("Segoe UI", 16, "bold"),
                fg=color,
                bg="white"
            )

            value.pack(anchor="w", pady=(5, 0))

            self.card_values[key] = value

        # Job results

        results_card = tk.Frame(
            content,
            bg="white",
            padx=15,
            pady=15
        )

        results_card.pack(
            fill="both",
            expand=True
        )

        tk.Label(
            results_card,
            text="Job Recommendations",
            font=("Segoe UI", 14, "bold"),
            bg="white",
            fg="#1e293b"
        ).pack(anchor="w", pady=(0, 10))

        columns = (
            "rank",
            "job",
            "match",
            "skill",
            "tfidf",
            "eligibility"
        )

        self.tree = ttk.Treeview(
            results_card,
            columns=columns,
            show="headings",
            height=5
        )

        headings = {
            "rank": "Rank",
            "job": "Job Position",
            "match": "Match %",
            "skill": "Skill %",
            "tfidf": "TF-IDF %",
            "eligibility": "Eligibility"
        }

        widths = {
            "rank": 55,
            "job": 180,
            "match": 100,
            "skill": 100,
            "tfidf": 100,
            "eligibility": 110
        }

        for col in columns:

            self.tree.heading(
                col,
                text=headings[col]
            )

            self.tree.column(
                col,
                width=widths[col],
                anchor="center"
            )

        self.tree.pack(fill="x")

        self.tree.bind(
            "<<TreeviewSelect>>",
            self.show_job_details
        )

        tk.Label(
            results_card,
            text="JOB DETAILS",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#475569"
        ).pack(anchor="w", pady=(15, 5))

        self.detail_box = ScrolledText(
            results_card,
            height=5,
            font=("Segoe UI", 9),
            bg="#f8fafc",
            relief="solid",
            bd=1,
            wrap="word"
        )

        self.detail_box.pack(fill="both", expand=True)

        self.detail_box.insert(
            "1.0",
            "Analyze a resume to see job recommendations and eligibility details."
        )

        self.detail_box.config(state="disabled")

        # Footer

        footer = tk.Frame(
            self.root,
            bg="#e2e8f0",
            padx=15,
            pady=9
        )

        footer.pack(fill="x")

        tk.Label(
            footer,
            text="BFS / DFS SKILL EXPLORER",
            font=("Segoe UI", 9, "bold"),
            bg="#e2e8f0",
            fg="#334155"
        ).pack(side="left", padx=5)

        self.skill1 = tk.Entry(footer, width=14)
        self.skill1.pack(side="left", padx=4)
        self.skill1.insert(0, "python")

        self.skill2 = tk.Entry(footer, width=14)
        self.skill2.pack(side="left", padx=4)
        self.skill2.insert(0, "django")

        self.make_button(
            footer,
            "BFS",
            self.check_bfs,
            "#7c3aed"
        ).pack(side="left", padx=5)

        self.category = ttk.Combobox(
            footer,
            values=list(ontology.keys()),
            width=13,
            state="readonly"
        )

        self.category.set("tech")
        self.category.pack(side="left", padx=8)

        self.make_button(
            footer,
            "DFS",
            self.check_dfs,
            "#0891b2"
        ).pack(side="left", padx=4)

    # ----------------------------------------------------------
    # PDF UPLOAD
    # ----------------------------------------------------------

    def upload_pdf(self):

        filepath = filedialog.askopenfilename(
            title="Select Resume PDF",
            filetypes=[
                ("PDF Files", "*.pdf"),
                ("All Files", "*.*")
            ]
        )

        if not filepath:
            return

        try:

            reader = PdfReader(filepath)

            if reader.is_encrypted:
                messagebox.showerror(
                    "PDF Error",
                    "This PDF is password protected."
                )
                return

            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

            if not text.strip():

                messagebox.showerror(
                    "No Text Found",
                    "This PDF does not contain extractable text.\n\n"
                    "It may be a scanned image. Please use a text-based "
                    "PDF or paste the resume text manually."
                )

                return

            self.resume_input.delete("1.0", tk.END)

            self.resume_input.insert("1.0", text)

            self.current_file = filepath

            filename = filepath.replace("\\", "/").split("/")[-1]

            self.file_label.config(
                text="Uploaded: " + filename,
                fg="#059669"
            )

            messagebox.showinfo(
                "Upload Successful",
                "Resume PDF uploaded successfully!\n"
                "Click Analyze Resume to continue."
            )

        except Exception as e:

            messagebox.showerror(
                "PDF Error",
                str(e)
            )

    # ----------------------------------------------------------
    # LOAD SAMPLE
    # ----------------------------------------------------------

    def load_sample(self):

        name = tk.simpledialog.askstring(
            "Sample Resume",
            "Enter sample number:\n"
            "1 - Riya Sharma\n"
            "2 - Aman Verma\n"
            "3 - Neha Gupta"
        )

        if name not in ["1", "2", "3"]:
            return

        candidate = list(sample_resumes.keys())[int(name) - 1]

        self.resume_input.delete("1.0", tk.END)

        self.resume_input.insert(
            "1.0",
            sample_resumes[candidate]
        )

        self.file_label.config(
            text="Sample loaded: " + candidate,
            fg="#2563eb"
        )

    # ----------------------------------------------------------
    # ANALYZE RESUME
    # ----------------------------------------------------------

    def analyze_resume(self):

        resume = self.resume_input.get(
            "1.0",
            tk.END
        ).strip()

        if not resume:

            messagebox.showwarning(
                "Missing Resume",
                "Please upload a PDF or enter resume text."
            )

            return

        try:

            info, ranked = match_jobs(resume)

            self.info = info
            self.ranked = ranked

            # Candidate cards

            self.card_values["candidate"].config(
                text=info["name"][:16]
            )

            self.card_values["cgpa"].config(
                text=str(info["cgpa"])
            )

            self.card_values["experience"].config(
                text=str(info["exp"]) + " Years"
            )

            self.card_values["skills"].config(
                text=str(len(info["skills"]))
            )

            # Clear old results

            for item in self.tree.get_children():
                self.tree.delete(item)

            # Insert new results

            for rank, (i, r) in enumerate(ranked, 1):

                final, s_score, t_score, eligible, reasons, missing = r

                self.tree.insert(
                    "",
                    "end",
                    iid=str(i),
                    values=(
                        rank,
                        jobs[i]["title"],
                        f"{final * 100:.1f}%",
                        f"{s_score * 100:.1f}%",
                        f"{t_score * 100:.1f}%",
                        "Eligible" if eligible else "Not Eligible"
                    )
                )

            # Display first job details

            if ranked:
                self.tree.selection_set(str(ranked[0][0]))
                self.show_job_details()

            messagebox.showinfo(
                "Analysis Complete",
                "Resume analyzed successfully!\n"
                "Job recommendations are ready."
            )

        except Exception as e:

            messagebox.showerror(
                "Analysis Error",
                str(e)
            )

    # ----------------------------------------------------------
    # DISPLAY JOB DETAILS
    # ----------------------------------------------------------

    def show_job_details(self, event=None):

        selected = self.tree.selection()

        if not selected:
            return

        i = int(selected[0])

        result = next(
            (r for job_id, r in self.ranked if job_id == i),
            None
        )

        if result is None:
            return

        final, s_score, t_score, eligible, reasons, missing = result

        job = jobs[i]

        report = []

        report.append("POSITION: " + job["title"])

        report.append(
            f"\nOverall Match: {final * 100:.1f}%"
        )

        report.append(
            f"Skill Match: {s_score * 100:.1f}%"
        )

        report.append(
            f"TF-IDF Similarity: {t_score * 100:.1f}%"
        )

        report.append(
            "\nEligibility: " +
            ("ELIGIBLE" if eligible else "NOT ELIGIBLE")
        )

        report.append("\nEligibility Rules:")

        report.extend(reasons)

        report.append(
            "\nRequired Skills: " +
            ", ".join(job["required"])
        )

        report.append(
            "\nMissing Skills: " +
            (", ".join(missing) if missing else "None")
        )

        self.detail_box.config(state="normal")

        self.detail_box.delete("1.0", tk.END)

        self.detail_box.insert(
            "1.0",
            "\n".join(report)
        )

        self.detail_box.config(state="disabled")

    # ----------------------------------------------------------
    # CLEAR ALL
    # ----------------------------------------------------------

    def clear_all(self):

        self.resume_input.delete("1.0", tk.END)

        self.file_label.config(
            text="No file selected",
            fg="#64748b"
        )

        self.current_file = ""

        self.info = None
        self.ranked = []

        for item in self.tree.get_children():
            self.tree.delete(item)

        for value in self.card_values.values():
            value.config(text="--")

        self.detail_box.config(state="normal")

        self.detail_box.delete("1.0", tk.END)

        self.detail_box.insert(
            "1.0",
            "Analyze a resume to see job recommendations."
        )

        self.detail_box.config(state="disabled")

    # ----------------------------------------------------------
    # BFS TOOL
    # ----------------------------------------------------------

    def check_bfs(self):

        a = self.skill1.get().strip().lower()
        b = self.skill2.get().strip().lower()

        if not a or not b:

            messagebox.showwarning(
                "Missing Skills",
                "Enter both skills."
            )

            return

        d = bfs_distance(a, b)

        if d == -1:

            messagebox.showinfo(
                "BFS Result",
                "These skills are not connected."
            )

        else:

            messagebox.showinfo(
                "BFS Result",
                f"{a} and {b} are {d} step(s) apart."
            )

    # ----------------------------------------------------------
    # DFS TOOL
    # ----------------------------------------------------------

    def check_dfs(self):

        cat = self.category.get()

        if cat not in ontology:
            return

        found = dfs_skills(cat)

        messagebox.showinfo(
            "DFS Result",
            "Skills under " + cat + ":\n\n" +
            ", ".join(found[1:])
        )


# ==============================================================
# 11. START APPLICATION
# ==============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = ResumeScreenerGUI(root)

    root.mainloop()
