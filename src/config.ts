export const siteConfig = {
  name: "Anesa Ibrahimi",
  title: "Artificial Intelligence Graduate",
  description: "Portfolio website of Anesa Ibrahimi",
  accentColor: "#1d4ed8",
  social: {
    email: "anesaibrahimi26@gmail.com",
    linkedin: "https://www.linkedin.com/in/anesa-ibrahimi/",
    // twitter: "https://x.com/rfitzio",
    github: "https://github.com/anesaibr",
    cv : "/Anesa_Ibrahimi_Personal_CV.pdf"
  },
  aboutMe:
    "Recent MSc Artificial Intelligence graduate (UvA) passionate about Machine Learning, Deep Learning, Computer Vision, Reinforcement Learning, and Robotics. I enjoy building end-to-end research prototypes and production-ready systems, from modeling to evaluation. Highlights: memory-augmented ViT thesis; multi-modal NAO robot interaction; arXiv publication on in-context learning for VLMs.",
  skills: [
    "Python",
    "SQL",
    "MATLAB",
    "PyTorch",
    "TensorFlow",
    "Keras",
    "Machine Learning",
    "Deep Learning",
    "Computer Vision",
    "Reinforcement Learning",
    "Foundational Models",
    "Natural Language Processing",
    "Data Analysis"
  ],
  projects: [
    {
      name: "Fine-grained image understanding with VLMs (Master Thesis)",
      description:
        "Designed a memory-augmented Vision Transformer by injecting trainable key-value memory into MLP layers and trained it via knowledge distillation from CLIP. Achieved SOTA-level gains on long-caption, fine-grained retrieval benchmarks. Ran on HPC (Slurm; A100/H100).",
      repo: "https://github.com/anesaibr/open_clip", 
      paper: "/Anesa_Ibrahimi_Thesis_AI_Final.pdf",
      skills: ["Python", "PyTorch", "Vision Transformers","Foundational Models","CLIP", "CUDA", "HPC"]
    },
    {
      name: "In-Context Learning Improves Compositional Understanding of VLMs",
      description:
        "Compared contrastive vs. generative VLMs; analyzed data/architectures; applied ICL to boost compositional reasoning. Published at ICML ‘Foundation Models in the Wild’ Workshop.",
      link: "https://arxiv.org/abs/2407.15487",
      repo: "https://github.com/HoeZey/vlm-compositionality",
      paper: "/ICL_ICML_paper.pdf",
      skills: ["Python", "VLMs", "ICL", "Evaluation", "Benchmarking"]
    },
    {
      name: "Multi-Modal NAO Robot (Socially Intelligent Robotics)",
      description:
        "Led a 7-member team to build real-time multi-modal emotion recognition and empathetic response for NAO, integrating speech, text, and vision models with emotional TTS (OpenVoice). Conducted user trials on empathy and satisfaction.",
      link: "", // add demo/video if available
      paper: "/SIR_Final_Report_Gr_21.pdf",
      skills: ["Python", "PyTorch", "HuggingFace", "OpenVoice", "Docker", "NAO"]
    },

    {
      name: "AI for Medical Imaging (AI4MI)",
      description:
        "Developed and evaluated deep learning models for automatic segmentation of thoracic organs at risk (OARs) in CT scans using the SegTHOR dataset. Implemented and compared CNN-based architectures including ENet, UNet, UNet++ and DeepLabV3+, optimizing loss functions to address class imbalance.",
      repo: "https://github.com/prundeanualin/ai4mi_project",
      paper: "/AI4MI_Final_Report_Gr_13.pdf",
      skills: [
        "Python",
        "PyTorch",
        "Medical Image Segmentation",
        "Deep Learning",
        "Computer Vision",
        "CNNs",
        "UNet",
        "Data Augmentation"
      ]
    },

    {
      name: "LLM4CS",
      description:
        "Reproduced and extended the LLM4CS framework to test reproducibility in conversational information retrieval. Verified prompting strategies with GPT-3.5, introduced sparse BM25 retrieval and the open-source Llama 3.1-8B model, and addressed major reproducibility challenges in dense retrieval pipelines.",
      repo: "https://github.com/angelbujalance/LLM4CS",
      paper: "/ACM_Conference_Proceedings_LLM4CS.pdf",
      skills: [
        "Python",
        "Pyserini",
        "Information Retrieval",
        "Large Language Models",
        "Conversational Search",
        "Dense & Sparse Retrieval",
        "Prompt Engineering"
      ]
    },

    {
      name: "Instance Diffusion — Extension",
      description:
        "Automated instance prompts and bounding boxes with an LLM to reduce manual inputs for instance-aware diffusion. Improved efficiency while maintaining alignment with original study metrics.",
      link: "",
      repo : "https://github.com/Jellemvdl/InstanceDiffusion-extension",
      skills: ["Python", "LLM", "Deep Learning", "Computer Vision" ,  "Diffusion Models"]
    },
    // {
    //   name: "LICO (FACT in AI)",
    //   description:
    //     "Reproduced a project integrating linguistic prompts with visual features to improve interpretability in image classification.",
    //   link: "",
    //   skills: ["Python", "NLP", "Explainability", "Jupyter"]
    // },



    {
      name: "Optimizing Locomotion with Evolutionary Algorithms in MuJoCo (BSc Thesis)",
      description:
        "Evaluated CMA-ES, XNES, and SNES on continuous control in the MuJoCo Ant environment. Built a full simulation + optimization pipeline across 100 environments; ran quantitative and qualitative benchmarks.",
      link: "",
      paper: "/VU_AI_Bachelor_Thesis_Anesa.pdf",
      demoLink: "/cmaes_700_0.4_0.1.mp4",
      skills: ["Python", "MuJoCo", "OpenAI Gym", "EvoTorch", "NumPy", "Matplotlib"]
    }
  ],

  experience: [
    {
      company: "Vrije Universiteit Amsterdam",
      title: "Professor's Assistant — Databases",
      dateRange: "Mar 2023 – May 2023",
      bullets: [
        "Planned and taught weekly practicals for relational database systems.",
        "Guided students in database modeling and SQL (search/update).",
        "Monitored and supported student progress."
      ]
    },
    {
      company: "Vrije Universiteit Amsterdam",
      title: "Professor's Assistant — Human-Computer Interaction",
      dateRange: "Mar 2022 – May 2022",
      bullets: [
        "Led weekly practicals on interactive systems principles and techniques.",
        "Coached students on low/high-fidelity prototyping and evaluation.",
        "Tracked student progress and provided feedback."
      ]
    }
  ],

  education: [
    {
      school: "Universiteit van Amsterdam",
      degree: "MSc Artificial Intelligence",
      dateRange: "2023 – 2025",
      achievements: []
    },
    {
      school: "Vrije Universiteit Amsterdam",
      degree: "BSc Artificial Intelligence",
      specialization: "Intelligent Systems",
      minor : "Data Science",
      dateRange: "2020 – 2023",
      achievements: []
    },
  ]
};
