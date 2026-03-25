#!/usr/bin/env python3
"""Collect per-token expert selections from ALL MoE layers via prefill.

Strategy: feed each topic as a single forward pass (prefill).  The MoE switch
fires once per layer with indices shape (seq_len, K), giving us token-level
selections without any generation loop.

Data layout
-----------
  .run/expert-data/{model_tag}/topic_{i:02d}_{name}.npz
    layer_0  : int16 (n_tok, K)   ← sorted selected expert indices
    layer_1  : int16 (n_tok, K)
    ...
    layer_N  : int16 (n_tok, K)
    meta_n_tokens    : scalar
    meta_train_n     : scalar   (first 80% = train)
    meta_test_n      : scalar   (last  20% = test)

  .run/expert-data/topics.json
    [{id, name, 35b_tokens, 122b_tokens}, ...]

Usage
-----
  poetry run python benchmarks/collect_expert_data.py \\
    --model  ~/.cache/.../Qwen3.5-35B-A3B-4bit/snapshots/... \\
    --index  .run/qwen35-35b-expert-index.json \\
    --tag    35b \\
    --out-dir .run/expert-data

  poetry run python benchmarks/collect_expert_data.py \\
    --model  ~/.cache/.../Qwen3.5-122B-A10B-4bit/snapshots/... \\
    --index  .run/qwen35-122b-expert-index.json \\
    --tag    122b \\
    --out-dir .run/expert-data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ── 10 topics, each ~800-1000 words ─────────────────────────────────────────

TOPICS: list[tuple[str, str]] = [
    ("quantum_physics", """
Quantum mechanics is the branch of physics that describes the behavior of matter and energy at the smallest scales,
where classical Newtonian physics breaks down entirely. At the quantum level, particles do not follow definite
trajectories but are instead described by wave functions, mathematical objects that encode the probability amplitudes
for all possible measurement outcomes. The Schrödinger equation governs how these wave functions evolve in time,
and it predicts interference, tunneling, and superposition — phenomena with no classical analogue.

Wave-particle duality lies at the heart of quantum theory. An electron fired at a double slit produces an
interference pattern on a detector screen, as though it travels as a wave through both slits simultaneously.
Yet if a detector is placed to identify which slit the electron passes through, the interference pattern
disappears and the electron behaves as a particle. This measurement problem — the collapse of the wave function
upon observation — remains one of the deepest puzzles in physics, leading to interpretations such as the Copenhagen
interpretation, many-worlds, and pilot-wave theory.

The Heisenberg uncertainty principle establishes a fundamental limit on the precision with which pairs of
complementary observables, such as position and momentum, can be simultaneously known. This is not a statement
about measurement disturbance but about the intrinsic nature of quantum states. A particle with a precisely
defined momentum is completely delocalized in space, and vice versa. The uncertainty principle has profound
implications for atomic structure: electrons cannot fall into the nucleus because that would require simultaneously
well-defined position and momentum, violating the principle.

Quantum entanglement arises when two particles interact such that their quantum states become correlated in a way
that persists regardless of the distance separating them. Measuring one particle instantly determines the
correlated property of the other, no matter how far apart they are. Einstein called this spooky action at a
distance and believed it indicated that quantum mechanics was incomplete. Bell's theorem and subsequent experiments
demonstrated that no local hidden variable theory can reproduce quantum mechanical predictions, confirming
the non-local nature of entanglement without allowing faster-than-light communication.

Particle physics describes the fundamental constituents of matter through the Standard Model, which categorizes
particles into fermions — quarks and leptons — and bosons, which mediate forces. Quarks combine via the strong
nuclear force, carried by gluons, to form hadrons such as protons and neutrons. The electromagnetic force is
mediated by photons, the weak nuclear force by W and Z bosons, and the Higgs boson gives mass to other particles
through the Higgs mechanism. The discovery of the Higgs boson at the Large Hadron Collider in 2012 completed
the Standard Model, though gravity remains outside its framework and dark matter and dark energy are unexplained.

Quantum field theory extends quantum mechanics to fields, treating particles as excitations of underlying fields
that permeate all of space. Quantum electrodynamics describes the interaction between charged particles and
photons with extraordinary precision — its predictions agree with experiment to more than ten decimal places.
Renormalization, the technique for handling infinities that arise in perturbative calculations, was a major
theoretical achievement. The path integral formulation, developed by Feynman, provides an alternative but
equivalent description where the probability amplitude for a process is a sum over all possible histories.

Quantum computing exploits superposition and entanglement to perform computations that would be intractable
for classical computers. Qubits can exist in superpositions of zero and one, and quantum gates manipulate
these superpositions coherently. Shor's algorithm can factor large integers exponentially faster than any
known classical algorithm, threatening current cryptographic systems. Grover's algorithm provides a quadratic
speedup for unstructured search. Building practical quantum computers requires overcoming decoherence —
the loss of quantum coherence due to environmental interactions — which currently limits qubit counts and
gate fidelities in systems based on superconducting circuits, trapped ions, and photonic platforms.
"""),

    ("ancient_rome", """
The Roman Republic emerged in 509 BCE when the Romans expelled their Etruscan kings and established a system
of government based on annually elected magistrates, a Senate, and popular assemblies. The Republic was defined
by its constitution — unwritten but powerful — which distributed power among consuls, praetors, censors, quaestors,
and tribunes, with the Senate as the deliberative backbone. The principle of collegiality, requiring two consuls
to govern simultaneously, and the limited one-year terms were designed to prevent any individual from accumulating
tyrannical power. This system, however imperfect, endured for centuries and shaped Western political thought.

Roman expansion began modestly in central Italy, driven by wars with neighboring peoples such as the Samnites,
Latins, and Etruscans. The conquest of the Italian peninsula by the third century BCE gave Rome access to large
military levies and agricultural surpluses. The Punic Wars against Carthage, spanning from 264 to 146 BCE,
were transformative. The First Punic War brought Sicily under Roman control. The Second, launched by Hannibal
Barca's audacious crossing of the Alps with war elephants, brought Carthaginian forces to the gates of Rome
itself. Hannibal's victories at Trebia, Lake Trasimene, and Cannae, where he annihilated a Roman army of
eighty thousand, remain case studies in military envelopment. Yet Rome's resilience, allied network, and
Scipio Africanus's campaign in North Africa ultimately prevailed.

The late Republic was marked by social conflict, land reform disputes, and the rise of populares and optimates
as factional labels. The Gracchi brothers, Tiberius and Gaius, attempted agrarian reforms to redistribute
public land to landless citizens, provoking aristocratic opposition that culminated in their violent deaths.
The Social War of 91-87 BCE resulted in Roman citizenship being extended to most Italian allies, fundamentally
altering the political landscape. Sulla's march on Rome with his legions set a precedent for military commanders
using their armies as political instruments, a pattern that would recur with Julius Caesar.

Julius Caesar's career encapsulated the contradictions of the late Republic. A brilliant general, he conquered
Gaul over nine years, enriching himself and his soldiers while narrating his campaigns in the Commentarii de
Bello Gallico with elegant precision. His crossing of the Rubicon in 49 BCE with the Thirteenth Legion was an
act of treason that plunged Rome into civil war. After defeating Pompey and his supporters, Caesar was appointed
dictator perpetuo. His assassination on the Ides of March in 44 BCE by a senatorial conspiracy led by Brutus
and Cassius did not restore the Republic but instead triggered another cycle of civil wars culminating in the
principate of Augustus.

Augustus Caesar, born Gaius Octavius, transformed Rome from republic to empire while carefully preserving
republican forms. He held tribunician power, imperium proconsulare, and the title princeps — first citizen.
By controlling the provinces with armies and the treasury, he held real power while the Senate retained
nominal dignity. The Pax Romana, the long peace he inaugurated, lasted for two centuries and witnessed
extraordinary prosperity, urbanization, and the codification of Roman law. The network of roads — over
eighty thousand kilometers — knit the empire together and facilitated trade, military movement, and administration.

The culture of Imperial Rome was syncretic, absorbing Greek philosophy, religion, and aesthetics. Virgil's
Aeneid created a foundational myth linking Rome to Troy. Horace perfected Latin lyric poetry. Livy wrote
the monumental history of Rome from its founding. Roman architecture — the Pantheon, Colosseum, aqueducts,
basilicas — reflected engineering ambition and civic grandeur. Roman law, codified under Justinian centuries
later, became the foundation of legal systems across Europe and the Americas.

The fall of the Western Roman Empire in 476 CE, when the Germanic chieftain Odoacer deposed the last emperor
Romulus Augustulus, was the culmination of processes — political fragmentation, economic strain, military
reliance on barbarian foederati, and plague — that had been developing for centuries. The Eastern Empire,
centered at Constantinople, survived for nearly another thousand years as the Byzantine Empire, preserving
Roman law and Greek learning until the Ottoman conquest in 1453.
"""),

    ("machine_learning", """
Machine learning is a subfield of artificial intelligence in which systems learn patterns from data rather
than following explicitly programmed rules. The fundamental paradigm — that useful behavior can be extracted
from examples — underlies applications ranging from image recognition and natural language processing to
scientific discovery and autonomous systems. Three main learning paradigms define the field: supervised
learning, where the model is trained on labeled input-output pairs; unsupervised learning, where the model
discovers structure in unlabeled data; and reinforcement learning, where an agent learns by interacting with
an environment to maximize cumulative reward.

Linear models provide the foundation for understanding more complex methods. Linear regression fits a
hyperplane to minimize the mean squared error between predictions and targets. Logistic regression extends
this to classification by applying a sigmoid function to produce probability estimates. Support vector machines
find the maximum-margin hyperplane separating classes, with kernel methods allowing nonlinear decision
boundaries. Regularization techniques such as L1 and L2 penalties prevent overfitting by penalizing model
complexity, balancing the bias-variance tradeoff that is central to all of machine learning theory.

Neural networks are universal function approximators composed of layers of artificial neurons. Each neuron
computes a weighted sum of its inputs and applies a nonlinear activation function. Stacking many layers
creates deep networks capable of learning hierarchical representations: early layers detect low-level features
such as edges, intermediate layers detect parts such as eyes and noses, and final layers recognize abstract
concepts such as faces. The backpropagation algorithm computes gradients of a loss function with respect to
all weights via the chain rule, enabling gradient descent to optimize the network parameters.

Convolutional neural networks exploit the spatial structure of images by applying learnable filters that
share weights across positions. Each convolutional layer learns to detect local patterns, pooling layers
reduce spatial dimensions, and fully connected layers map features to outputs. AlexNet's 2012 ImageNet
victory demonstrated that deep CNNs trained on GPUs could dramatically outperform hand-engineered features
on large-scale image classification. Subsequent architectures — VGG, ResNet, Inception, EfficientNet — pushed
accuracy further through depth, skip connections, and neural architecture search.

Recurrent neural networks process sequential data by maintaining hidden states that evolve over time.
Long short-term memory networks address the vanishing gradient problem with gating mechanisms that selectively
remember and forget information. Gated recurrent units offer a simpler alternative. These architectures enabled
breakthroughs in speech recognition, machine translation, and language modeling, though they struggle with
very long sequences due to sequential computation bottlenecks.

The transformer architecture, introduced in Attention Is All You Need, replaced recurrence with self-attention,
allowing parallel computation over entire sequences. The attention mechanism computes weighted combinations of
value vectors, with weights determined by the similarity between query and key vectors. Multi-head attention
allows the model to attend to multiple aspects of the input simultaneously. Positional encodings inject
sequence order information. Transformers became the dominant architecture for language, then vision, and
eventually almost all modalities, enabling models such as GPT, BERT, T5, and their successors.

Mixture-of-experts models scale transformer capacity without proportionally increasing computation. Instead
of activating all parameters for every token, a routing network selects a small subset of expert networks
to process each input. This conditional computation allows models with hundreds of billions of parameters
to be trained and deployed at costs closer to smaller dense models. GShard, Switch Transformer, and GLaM
demonstrated that sparse MoE could match or exceed dense models of the same computational budget.
The routing mechanism — typically top-K selection from a learned gate — introduces load balancing challenges
that require auxiliary losses to prevent expert collapse.

Training stability and optimization are critical concerns in deep learning. Batch normalization and layer
normalization stabilize training by normalizing activations. Adam and its variants adapt learning rates
per parameter using gradient moments. Learning rate schedules — warmup followed by cosine decay — improve
convergence. Gradient clipping prevents exploding gradients. Mixed precision training with bfloat16 reduces
memory and increases throughput. These engineering advances, combined with scale, drove the empirical
progress that characterizes modern deep learning.
"""),

    ("calculus", """
Calculus is the mathematical study of continuous change, built on two foundational operations: differentiation
and integration, unified by the fundamental theorem. Developed independently by Newton and Leibniz in the
seventeenth century, calculus provided the mathematical language for classical mechanics, electromagnetism,
fluid dynamics, and a vast range of other scientific and engineering disciplines. The modern rigorous
formulation, due to Cauchy, Weierstrass, and Riemann, replaced intuitive notions of infinitesimals with
precise epsilon-delta definitions of limits.

The derivative of a function measures its instantaneous rate of change. Formally, the derivative f′(x) is
the limit of the difference quotient (f(x+h)−f(x))/h as h approaches zero, provided this limit exists.
Geometrically, the derivative gives the slope of the tangent line to the graph at each point. The rules
of differentiation — product rule, quotient rule, chain rule — allow derivatives of complex functions to
be computed systematically. Higher derivatives describe acceleration, curvature, and the concavity of
functions, and they appear naturally in Taylor series, which approximate functions locally by polynomials.

Integration is the inverse operation of differentiation and represents accumulation or area under a curve.
The definite integral of f over an interval [a,b] is defined as the limit of Riemann sums, partitioning
the interval into subintervals and summing the products of function values and widths. The fundamental
theorem of calculus establishes that differentiation and integration are inverse processes: the derivative
of the integral of f with respect to its upper limit equals f, and the definite integral can be computed
by finding an antiderivative and evaluating at the endpoints. Techniques of integration include substitution,
integration by parts, partial fractions, and trigonometric substitution.

Multivariable calculus extends these ideas to functions of several variables. Partial derivatives measure
the rate of change with respect to one variable while holding others fixed. The gradient vector, composed
of all partial derivatives, points in the direction of steepest ascent and is fundamental to optimization.
The directional derivative measures the rate of change in an arbitrary direction as the dot product of
the gradient and the unit direction vector. Critical points, where the gradient vanishes, are candidates
for local extrema or saddle points, classified by the second derivative test involving the Hessian matrix.

Multiple integrals extend integration to regions in two or three dimensions. Double integrals compute
volumes under surfaces and can be evaluated as iterated integrals using Fubini's theorem when the integrand
is continuous. Change of variables via the Jacobian determinant allows integrals over complex regions to
be transformed to simpler ones — for example, switching to polar coordinates for regions with circular symmetry.
Triple integrals extend this to three dimensions, with spherical and cylindrical coordinates often simplifying
calculations of volumes, masses, and moments.

Vector calculus introduces differentiation operators for vector fields. The gradient maps scalar fields
to vector fields. The divergence measures the net flux of a vector field out of an infinitesimal volume
and appears in the continuity equation for fluid flow and Gauss's law for electric fields. The curl measures
the rotation of a vector field and appears in Faraday's law and the Navier-Stokes equations. The Laplacian,
the divergence of the gradient, governs diffusion, wave propagation, and electrostatics.

The major integral theorems unify these operators. Green's theorem relates a line integral around a closed
curve to a double integral over the enclosed region. Stokes' theorem generalizes this to three dimensions,
relating a surface integral of the curl to a line integral around the boundary curve. The divergence theorem
— also called Gauss's theorem — relates the flux integral of a vector field through a closed surface to the
volume integral of the divergence over the enclosed region. These theorems appear throughout physics as
conservation laws and form the mathematical foundation of Maxwell's equations of electromagnetism.

Ordinary differential equations describe how quantities change in time or space. A first-order linear ODE
can be solved via an integrating factor. Separable equations allow the variables to be separated and
integrated independently. The method of undetermined coefficients and variation of parameters solve
non-homogeneous linear ODEs. Systems of linear ODEs are solved using eigenvalues and eigenvectors of
the coefficient matrix. Qualitative analysis — phase portraits, stability of equilibria, Lyapunov functions —
provides insight when closed-form solutions are unavailable. Numerical methods such as Euler's method,
Runge-Kutta, and adaptive step-size algorithms enable practical simulation of complex systems.
"""),

    ("cell_biology", """
The cell is the fundamental unit of life, the smallest entity capable of carrying out the biochemical
processes that define living organisms. All known life forms are composed of cells, which range from the
simple prokaryotes — bacteria and archaea — lacking a membrane-bound nucleus, to the complex eukaryotes
including protists, fungi, plants, and animals. Cell theory, established by Schleiden, Schwann, and Virchow
in the nineteenth century, holds that all organisms are made of cells and that new cells arise only from
pre-existing cells. This principle underpins all of biology.

The cell membrane, a phospholipid bilayer with embedded proteins, forms the boundary between the cell
and its environment. The hydrophobic fatty acid tails face inward while the hydrophilic phosphate heads
face the aqueous environment on each side. This selective barrier controls the movement of ions, nutrients,
and waste products. Transport proteins — channels, carriers, and pumps — mediate the passage of specific
molecules. The sodium-potassium ATPase maintains electrochemical gradients critical for nerve impulse
propagation and nutrient uptake. Membrane receptors detect extracellular signals and initiate intracellular
signaling cascades.

The nucleus houses the cell's genetic information in the form of DNA organized into chromosomes. The nuclear
envelope, a double membrane punctuated by nuclear pores, regulates the import of proteins and the export
of RNA. Within the nucleus, the nucleolus is the site of ribosomal RNA synthesis. DNA replication begins
at multiple origins of replication, proceeds bidirectionally via DNA polymerase, and requires a suite of
accessory proteins including helicase, primase, ligase, and topoisomerase. The accuracy of replication is
maintained by proofreading activity and mismatch repair mechanisms, achieving error rates below one per
billion bases replicated.

Gene expression begins with transcription, in which RNA polymerase synthesizes a messenger RNA copy of
the DNA template strand. In eukaryotes, the primary transcript undergoes processing: the 5′ cap and poly-A
tail are added, and introns are removed by the spliceosome while exons are joined. Regulation of transcription
is mediated by transcription factors that bind specific DNA sequences and by chromatin remodeling that
alters the accessibility of DNA. Epigenetic modifications — DNA methylation and histone acetylation,
methylation, and phosphorylation — affect gene expression without changing the DNA sequence and can be
heritable through cell division.

Translation occurs at ribosomes, large ribonucleoprotein complexes that read the mRNA codon by codon
and catalyze peptide bond formation between amino acids brought by transfer RNAs. The ribosome has three
tRNA binding sites: A, P, and E. Initiation involves assembly of the ribosome at the start codon AUG.
Elongation adds amino acids sequentially. Termination occurs when a stop codon is encountered and release
factors trigger peptide release. Newly synthesized proteins often require folding assistance from molecular
chaperones and post-translational modifications such as glycosylation, phosphorylation, and ubiquitination.

Mitochondria are the primary sites of cellular energy production via oxidative phosphorylation. Pyruvate
from glycolysis is converted to acetyl-CoA, which enters the citric acid cycle, generating NADH and FADH2.
These electron carriers donate electrons to the respiratory chain in the inner mitochondrial membrane,
driving proton translocation across the membrane. The proton gradient drives ATP synthase to produce ATP
from ADP and inorganic phosphate. This process yields approximately 30-32 ATP per glucose molecule,
vastly more than the 2 ATP from glycolysis alone. Mitochondria also regulate apoptosis, calcium homeostasis,
and reactive oxygen species production.

Cell division ensures growth, repair, and reproduction. Mitosis distributes identical chromosomes to
daughter cells through prophase, metaphase, anaphase, and telophase. The spindle apparatus of microtubules,
nucleated at centrosomes, attaches to kinetochores on chromosomes and segregates them. The cell cycle is
regulated by cyclin-dependent kinases and their cyclin partners, with checkpoints ensuring DNA integrity
before division proceeds. Meiosis, the specialized division producing gametes, involves two rounds of
division and crossing over between homologous chromosomes, generating genetic diversity essential for
evolution. Errors in cell cycle regulation underlie cancer, which arises from unchecked proliferation
driven by mutations in oncogenes and tumor suppressor genes.
"""),

    ("ethics_philosophy", """
Ethics, the branch of philosophy concerned with questions of right and wrong, good and bad, virtue and vice,
has been central to human intellectual life since antiquity. The ancient Greeks established the foundational
questions: What is the good life? What makes an action right or wrong? What virtues should we cultivate?
Three major ethical frameworks have dominated Western moral philosophy: virtue ethics, originating with
Aristotle; deontological ethics, associated with Kant; and consequentialism, developed by Bentham and Mill.
Each captures genuine moral insights while facing serious objections.

Virtue ethics focuses on the character of the moral agent rather than on rules or outcomes. Aristotle argued
that human flourishing — eudaimonia, often translated as happiness or well-being — is achieved through the
exercise of virtues, stable dispositions to act, feel, and reason well in appropriate circumstances. Virtues
such as courage, justice, temperance, and practical wisdom occupy a mean between excess and deficiency —
courage lies between cowardice and recklessness. Virtuous action requires not merely doing the right thing
but doing it for the right reasons, with appropriate feeling, at the right time. Neo-Aristotelian virtue ethics,
revived by Anscombe, Foot, and MacIntyre, has flourished as an alternative to rule-based approaches.

Kant's deontological ethics grounds morality in reason rather than consequences or character. The categorical
imperative, Kant's supreme moral principle, commands us to act only according to maxims we could consistently
will to be universal laws. Lying fails this test because if everyone lied, the institution of truth-telling
on which lying depends would collapse. A second formulation commands treating rational beings always as ends
in themselves and never merely as means — a principle that underlies modern conceptions of human dignity
and rights. Kantian ethics demands strict adherence to duty regardless of consequences, which critics argue
leads to counterintuitive conclusions in extreme cases.

Consequentialism judges actions solely by their outcomes. Classical utilitarianism, formulated by Bentham
and refined by Mill, holds that the right action is the one that produces the greatest happiness for the
greatest number. This impartial aggregation of welfare has intuitive appeal and drives much of public policy
analysis. Preference utilitarianism replaces hedonic welfare with preference satisfaction. Rule utilitarianism
avoids some counterintuitive implications of act utilitarianism by judging actions by reference to rules
whose general acceptance would maximize utility. Critics argue that consequentialism can justify intuitively
wrong acts — punishing an innocent person to prevent greater harm, for instance — if the numbers work out.

Metaethics investigates the nature and status of moral claims rather than their content. Moral realism holds
that there are objective moral facts, independent of what anyone believes or feels. Cornell realism identifies
moral properties with natural properties; non-naturalism, associated with Moore, holds that goodness is a
simple, indefinable, non-natural property known by moral intuition. Error theory, defended by Mackie,
accepts that moral discourse makes objective claims but denies that any of them are true. Expressivism denies
that moral statements are truth-apt at all, construing them instead as expressions of attitudes or prescriptions.

Political philosophy applies ethical principles to questions of social organization. Rawls's theory of justice
grounds principles of fair distribution in what rational agents would choose behind a veil of ignorance,
not knowing their place in society. The difference principle permits inequalities only if they benefit the
worst-off members. Nozick's libertarian response defends individual rights against redistributive schemes,
arguing that free exchange of justly acquired holdings is sufficient for a just distribution. Communitarian
critics challenge both Rawls and Nozick for presupposing an unencumbered self abstracted from social context.
Feminist ethics emphasizes care, relationships, and context over abstract principles.

Applied ethics extends moral reasoning to specific domains. Bioethics addresses organ allocation, end-of-life
decisions, genetic enhancement, and research ethics. Environmental ethics asks whether non-human entities
have moral standing. Business ethics examines fiduciary duties, corporate social responsibility, and the
ethics of competition. The ethics of artificial intelligence raises novel questions about algorithmic bias,
accountability, autonomous weapons, and the moral status of sophisticated AI systems. These applied domains
test and refine our theoretical frameworks, demonstrating that ethics is not merely academic but urgently
practical.
"""),

    ("economics", """
Economics is the study of how individuals, firms, and societies allocate scarce resources to satisfy unlimited
wants. The field divides into microeconomics, which examines individual decision-making and market interactions,
and macroeconomics, which studies aggregate phenomena such as growth, inflation, unemployment, and monetary
and fiscal policy. The discipline has become increasingly mathematical, employing optimization theory, game
theory, statistical econometrics, and computational modeling to derive and test predictions about economic
behavior.

Supply and demand are the foundational analytical tools of microeconomics. The demand curve captures the
inverse relationship between price and quantity demanded, reflecting consumers' diminishing marginal utility.
The supply curve captures the positive relationship between price and quantity supplied, reflecting producers'
increasing marginal cost. In a competitive market, the price adjusts until quantity supplied equals quantity
demanded, achieving equilibrium. Comparative statics analyzes how changes in underlying factors — tastes,
income, production costs, taxes — shift the curves and alter equilibrium prices and quantities.

Consumer theory models individuals as utility maximizers subject to a budget constraint. Utility functions
represent preferences; indifference curves connect bundles providing equal utility. The budget line represents
the set of affordable bundles at given prices and income. The consumer optimum lies at the tangency between
the highest attainable indifference curve and the budget line, where the marginal rate of substitution equals
the price ratio. The Slutsky equation decomposes the price effect into substitution and income effects,
distinguishing normal goods from Giffen goods. Revealed preference theory reconstructs preferences from
observed choices without requiring utility functions.

Producer theory analyzes firms as profit maximizers. Production functions map inputs to outputs, with
isoquants playing the role of indifference curves. Cost minimization determines the optimal input mix at
given factor prices. Short-run costs distinguish fixed and variable components; the U-shaped average cost
curve reflects returns to scale. In the long run, all inputs are variable and firms enter or exit the
industry until economic profits are zero in competitive equilibrium. Monopoly firms face the entire demand
curve and choose output where marginal revenue equals marginal cost, producing less and charging more than
the competitive outcome, generating deadweight loss.

Game theory analyzes strategic interactions where payoffs depend on multiple agents' choices. The prisoner's
dilemma illustrates why individually rational choices can produce collectively suboptimal outcomes —
mutual defection dominates mutual cooperation even though both prefer cooperation. Nash equilibrium, a profile
of strategies where no player can improve by deviating unilaterally, is the central solution concept.
Repeated games allow cooperation to emerge via trigger strategies that punish defection. Mechanism design —
the reverse of game theory — asks what institutions or rules would implement a desired outcome as an
equilibrium, addressing auction design, matching markets, and incentive compatibility.

Macroeconomics grapples with output fluctuations, inflation, and the determinants of long-run growth.
The IS-LM model of Hicks characterizes short-run equilibrium as the intersection of goods market equilibrium
and money market equilibrium. The AD-AS model adds price adjustment. New Keynesian models introduce
nominal rigidities — sticky prices and wages — that generate real effects of monetary policy in the short run.
Real Business Cycle theory emphasizes technology shocks and flexible prices. Dynamic stochastic general
equilibrium models integrate microeconomic foundations with macroeconomic dynamics and are used by central
banks for policy analysis.

Monetary policy controls the money supply and interest rates to stabilize inflation and employment.
The Taylor rule describes how central banks adjust nominal interest rates in response to deviations of
inflation from target and output from potential. Quantitative easing purchases long-term assets to lower
long-term rates when the policy rate is at the zero lower bound. Fiscal policy uses government spending
and taxation to stabilize aggregate demand, subject to intertemporal budget constraints and Ricardian
equivalence concerns. International economics addresses trade — comparative advantage, gains from trade,
the Heckscher-Ohlin theorem — and finance — exchange rate determination, capital flows, and the trilemma
between fixed exchange rates, monetary independence, and capital mobility.
"""),

    ("climate_ecology", """
Earth's climate system is a complex, coupled interaction among the atmosphere, oceans, cryosphere, land
surface, and biosphere. Energy from the Sun drives this system, with incoming shortwave radiation balanced
by outgoing longwave radiation. The greenhouse effect arises because greenhouse gases — water vapor, carbon
dioxide, methane, nitrous oxide, and ozone — absorb outgoing infrared radiation and re-emit it in all
directions, warming the surface beyond what blackbody calculations would suggest for a planet at Earth's
distance from the Sun. Without the natural greenhouse effect, Earth's average surface temperature would be
approximately minus eighteen degrees Celsius rather than the observed plus fifteen.

The carbon cycle connects the reservoirs of carbon in the atmosphere, biosphere, soils, oceans, and
lithosphere through fluxes of exchange. Photosynthesis absorbs carbon dioxide from the atmosphere into
living biomass. Respiration and decomposition return it. The ocean absorbs roughly a quarter of anthropogenic
carbon dioxide emissions through dissolution into seawater and biological uptake by phytoplankton. Oceanic
carbon is stored in surface waters, transferred to deep water via the biological pump as organic particles
sink, and eventually buried in sediments. The slow geological carbon cycle, involving volcanic outgassing
and carbonate weathering, operates on million-year timescales.

Climate feedbacks amplify or dampen initial perturbations. The water vapor feedback is strongly positive:
warming increases atmospheric water vapor, which is itself a greenhouse gas. The ice-albedo feedback is
also positive: melting ice and snow expose darker ocean and land surfaces that absorb more solar radiation.
The lapse rate feedback is negative at low latitudes but positive at high latitudes. Cloud feedbacks are
complex and uncertain, differing between low marine clouds and high cirrus clouds. The climate sensitivity —
the equilibrium warming per doubling of carbon dioxide — is estimated at 2.5 to 4.0 degrees Celsius, with
the uncertainty driven primarily by cloud and aerosol interactions.

Ecological systems are structured by energy flow and nutrient cycling. Primary producers — plants, algae,
and photosynthetic bacteria — convert solar energy into organic matter. Herbivores consume primary producers,
carnivores consume herbivores, and decomposers break down dead organic matter, returning nutrients to the
soil and water. The ten percent rule states that roughly ten percent of energy at each trophic level is
transferred to the next, limiting food chain length. Biogeochemical cycles — the nitrogen, phosphorus, and
sulfur cycles — govern the availability of essential nutrients and are profoundly affected by agricultural
practices, industrial emissions, and land use change.

Biodiversity encompasses the variety of life at genetic, species, and ecosystem levels. Hotspots of
biodiversity — tropical rainforests, coral reefs, Mediterranean-climate shrublands — harbor disproportionate
fractions of the world's species. Ecological communities are organized by competitive exclusion, niche
partitioning, predator-prey dynamics, and mutualistic relationships such as pollination and mycorrhizal
associations. Keystone species exert disproportionate influence on community structure relative to their
abundance. Invasive species — organisms introduced to regions outside their native range — disrupt
established communities and drive native species to extinction, constituting one of the leading drivers
of biodiversity loss.

Climate change is altering ecological systems globally. Species ranges are shifting poleward and to higher
elevations as temperatures rise. Phenological mismatches — shifts in the timing of migration, flowering,
and insect emergence — disrupt predator-prey and plant-pollinator relationships. Coral bleaching events,
driven by sea surface temperature anomalies, have devastated reef ecosystems in the Indian Ocean, Pacific,
and Caribbean. Permafrost thaw in Arctic regions releases methane and carbon dioxide stored for millennia,
constituting a positive feedback. Sea level rise from thermal expansion and ice sheet melt threatens coastal
and island ecosystems and human settlements.

Sustainable development seeks to meet present needs without compromising the ability of future generations
to meet their own needs. Ecosystem services — provisioning services such as food and water, regulating
services such as carbon sequestration and flood control, cultural services, and supporting services such
as nutrient cycling and soil formation — provide the material basis for human welfare. Economic valuation
of ecosystem services, through methods such as contingent valuation, hedonic pricing, and production
function approaches, aims to incorporate natural capital into decision-making. Protected areas, sustainable
agriculture, restoration ecology, and international agreements on biodiversity and climate are instruments
for managing human impact on the biosphere.
"""),

    ("medicine_anatomy", """
Human anatomy and physiology describe the structure and function of the body's organ systems, from the
cellular level to the whole organism. The body is organized hierarchically: atoms combine into molecules,
molecules into organelles, organelles into cells, cells into tissues, tissues into organs, and organs into
systems. The major systems — integumentary, skeletal, muscular, nervous, endocrine, cardiovascular, lymphatic,
respiratory, digestive, urinary, and reproductive — are interdependent, and understanding their integration
is essential for clinical medicine.

The cardiovascular system pumps blood throughout the body, delivering oxygen and nutrients to tissues and
removing carbon dioxide and metabolic waste. The heart is a four-chambered pump divided into right and
left sides by the interventricular septum. The right side receives deoxygenated blood from the systemic
circulation and pumps it through the pulmonary circulation to the lungs for oxygenation. The left side
receives oxygenated blood from the lungs and pumps it through the aorta into the systemic circulation.
Cardiac output — the product of stroke volume and heart rate — is regulated by the autonomic nervous system,
Frank-Starling mechanisms, and circulating catecholamines to match metabolic demands.

The respiratory system facilitates gas exchange between the atmosphere and the blood. Air enters through
the nose and mouth, is warmed and humidified, and passes through the larynx, trachea, and branching bronchi
to the alveoli — the terminal air sacs where gas exchange occurs across a thin epithelial layer into pulmonary
capillaries. Ventilation is driven by the diaphragm and intercostal muscles creating negative intrathoracic
pressure. The partial pressure gradients of oxygen and carbon dioxide drive diffusion. Hemoglobin in red
blood cells binds oxygen cooperatively — the sigmoidal oxygen-hemoglobin dissociation curve reflects this
cooperativity and allows efficient loading at the lungs and unloading in peripheral tissues.

The nervous system processes sensory information and coordinates motor responses. The central nervous system
comprises the brain and spinal cord. The cerebral cortex, organized into frontal, parietal, temporal, and
occipital lobes, mediates higher cognitive functions, sensation, and voluntary movement. The limbic system
regulates emotion and memory. The cerebellum coordinates movement and balance. The brainstem controls vital
functions including respiration and heart rate. The peripheral nervous system includes somatic nerves
mediating voluntary sensation and movement, and the autonomic nervous system mediating involuntary control
of visceral organs. Neurons communicate via electrical action potentials and chemical neurotransmitters
at synapses.

The endocrine system uses hormones — chemical messengers secreted into the blood — to coordinate physiology
over longer timescales than neural signaling. The hypothalamic-pituitary axis integrates the nervous and
endocrine systems, with the hypothalamus releasing neuropeptides that regulate pituitary hormone secretion.
The pituitary controls thyroid, adrenal cortex, and gonadal function. Insulin and glucagon from the pancreatic
islets regulate blood glucose. Cortisol from the adrenal cortex mediates stress responses and suppresses
inflammation. Thyroid hormones regulate metabolic rate and are essential for development. Feedback loops
maintain homeostasis by suppressing hormone secretion when target organ effects reach appropriate levels.

The immune system defends against infection and malignancy through innate and adaptive branches. Innate
immunity provides rapid, non-specific defense via physical barriers, phagocytes, natural killer cells,
and the complement system. Pattern recognition receptors such as Toll-like receptors detect conserved
microbial structures and activate inflammatory responses. Adaptive immunity provides specific, long-lasting
protection. B lymphocytes produce antibodies that neutralize pathogens or mark them for destruction.
T lymphocytes coordinate adaptive responses through helper functions and directly kill infected or malignant
cells. Immunological memory, mediated by long-lived memory B and T cells, is the basis of vaccination.

Pathophysiology describes how disease disrupts normal function. Atherosclerosis involves lipid deposition,
endothelial dysfunction, and inflammatory cell recruitment in arterial walls, narrowing the lumen and
risking ischemia or infarction. Diabetes results from insufficient insulin secretion or peripheral insulin
resistance, leading to hyperglycemia and long-term microvascular and macrovascular complications.
Cancer arises from accumulated somatic mutations activating oncogenes and inactivating tumor suppressors,
enabling unlimited proliferation, evasion of apoptosis, angiogenesis, and metastasis. Treatment advances —
targeted therapies, immunotherapy, precision medicine guided by genomic profiling — are transforming
oncology and shifting survival statistics across many cancer types.
"""),

    ("programming_systems", """
Computer science encompasses the theory, design, and implementation of computational systems, from the
mathematical foundations of algorithms and complexity to the engineering of operating systems, databases,
and distributed networks. The field divides broadly into theoretical computer science, which studies
computation abstractly, and applied areas including software engineering, computer architecture, artificial
intelligence, and systems programming.

Algorithms are step-by-step procedures for solving computational problems, and their analysis focuses on
time and space complexity — how resource requirements grow with input size. The asymptotic notation of
big-O, big-omega, and big-theta characterizes worst-case, best-case, and average-case complexity.
Sorting algorithms illustrate the tradeoffs: insertion sort is simple and fast for small inputs but O(n²)
in the worst case; merge sort guarantees O(n log n) but requires O(n) extra space; quicksort achieves
O(n log n) expected time with in-place sorting but O(n²) worst case. The comparison-based sorting lower
bound of Omega(n log n) follows from an information-theoretic argument about decision trees.

Data structures organize data to support efficient operations. Arrays provide O(1) random access.
Linked lists allow O(1) insertion and deletion but O(n) lookup. Hash tables achieve expected O(1) for
insert, delete, and search via hash functions and collision resolution. Binary search trees maintain
sorted order and support O(log n) operations when balanced; red-black trees and AVL trees guarantee
balance through rotations. Heaps implement priority queues with O(log n) insert and extract-min.
Graphs represent relationships between objects; adjacency lists and matrices trade memory for access
time. The choice of data structure profoundly affects algorithm performance and code complexity.

Operating systems manage hardware resources and provide abstractions for application programs. Process
management creates the illusion of concurrent execution via context switching and scheduling algorithms
— round-robin, priority-based, completely fair scheduling. Virtual memory maps process address spaces
to physical memory using page tables and the translation lookaside buffer, allowing each process to
believe it has exclusive access to a large address space while the OS manages physical allocation,
demand paging, and page replacement. File systems organize persistent storage into hierarchies of
directories and files; inodes store metadata while extent trees or B-trees manage data blocks.
Synchronization primitives — mutexes, semaphores, condition variables — coordinate concurrent access
to shared resources, preventing race conditions and deadlocks.

Computer networks enable communication between distributed systems. The OSI model layers physical,
data link, network, transport, session, presentation, and application protocols. TCP/IP dominates
modern networking: IP provides best-effort packet delivery with routing through autonomous systems
using BGP; TCP provides reliable, ordered delivery through sequence numbers, acknowledgments, and
retransmission. The three-way handshake establishes connections. Flow control and congestion control —
slow start, additive increase multiplicative decrease, fast retransmit — prevent network overload.
HTTP and HTTPS, built on TCP, serve as the backbone of the web. TLS provides encryption, authentication,
and integrity through asymmetric key exchange and symmetric encryption.

Databases store and query structured data reliably and efficiently. Relational databases organize data
into tables with schemas, enforce integrity constraints, and support SQL queries with joins, aggregations,
and subqueries. The ACID properties — atomicity, consistency, isolation, durability — ensure that
transactions are processed reliably even in the presence of failures. Indexes, B-tree and hash-based,
accelerate query execution. Query optimizers choose execution plans by estimating costs. NoSQL databases —
document stores, key-value stores, column stores, and graph databases — sacrifice some SQL flexibility
for horizontal scalability and flexible schemas, trading consistency for availability according to the
CAP theorem.

Compilers translate source code into machine code through a pipeline of lexical analysis, parsing,
semantic analysis, intermediate code generation, optimization, and code generation. Lexers tokenize
the source; parsers build abstract syntax trees according to a grammar. Type checkers enforce semantic
constraints. Intermediate representations such as SSA form enable dataflow analyses — reaching definitions,
liveness, available expressions — that support optimizations including constant folding, dead code
elimination, inlining, and loop transformations. Register allocation maps virtual registers to physical
registers, solving a graph coloring problem. Just-in-time compilation, used in Java, JavaScript, and
Python runtimes, compiles hot code paths at runtime, combining the flexibility of interpretation with
the performance of compilation.
"""),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--native-reader", required=True)
    p.add_argument("--tag", required=True, help="e.g. 35b or 122b")
    p.add_argument("--out-dir", default=".run/expert-data")
    p.add_argument("--max-tokens", type=int, default=1000,
                   help="Max tokens per topic (default 1000)")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--component-workers", type=int, default=3)
    p.add_argument("--topics", type=int, nargs="*", default=None,
                   help="Topic indices to collect (default: all 10)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    from streaming_moe.runtime import build_streamed_model, set_routed_top_k, _get_moe_module
    from streaming_moe.streamed_switch import StreamedSwitchGLU
    from streaming_moe.prefetch_switch import PrefetchingStreamedSwitchGLU
    from mlx_lm.models.cache import make_prompt_cache

    top_k = args.top_k
    topic_indices = args.topics if args.topics is not None else list(range(len(TOPICS)))

    print(f"Loading model [{args.tag}] …", flush=True)
    model, tokenizer, _, _ = build_streamed_model(
        Path(args.model), Path(args.index),
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
    )
    set_routed_top_k(model, top_k)

    # Discover all MoE layer indices
    tm = getattr(getattr(model, "language_model", model), "model", model)
    all_moe_layers: list[int] = []
    for i, layer in enumerate(tm.layers):
        moe, _ = _get_moe_module(layer)
        if moe is not None and getattr(moe, "switch_mlp", None) is not None:
            all_moe_layers.append(i)
    print(f"MoE layers: {len(all_moe_layers)}  ({all_moe_layers[0]}–{all_moe_layers[-1]})", flush=True)

    # ── Recording hook (handles both single-token and multi-token) ──
    selections: dict[int, list[np.ndarray]] = {}  # layer → list of (K,) arrays

    class _Hook(nn.Module):
        def __init__(self, inner, layer_idx: int):
            super().__init__()
            self._inner = inner
            self._l = layer_idx

        def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
            idx = np.array(indices.tolist(), dtype=np.int16)
            idx = idx.reshape(-1, top_k)        # (n_tok, K) in all cases
            for row in idx:
                selections[self._l].append(row.copy())
            return self._inner(x, indices)

    # Install hooks on all MoE layers
    for i, layer in enumerate(tm.layers):
        if i not in all_moe_layers:
            continue
        moe, _ = _get_moe_module(layer)
        sw = getattr(moe, "switch_mlp", None)
        if sw and isinstance(sw, (StreamedSwitchGLU, PrefetchingStreamedSwitchGLU)):
            moe.switch_mlp = _Hook(sw, i)

    topic_meta = []

    for topic_idx in topic_indices:
        name, text = TOPICS[topic_idx]
        out_path = out_dir / f"topic_{topic_idx:02d}_{name}.npz"

        if out_path.exists():
            print(f"  [skip] topic {topic_idx:02d} {name} already exists", flush=True)
            existing = np.load(out_path)
            n = int(existing["meta_n_tokens"])
            topic_meta.append({"id": topic_idx, "name": name, f"{args.tag}_tokens": n,
                                "train_n": int(existing["meta_train_n"]),
                                "test_n": int(existing["meta_test_n"])})
            continue

        # Tokenize (strip leading whitespace, cap at max_tokens)
        tokens = tokenizer.encode(text.strip())[:args.max_tokens]
        n_tok = len(tokens)
        print(f"  Topic {topic_idx:02d} [{name}]: {n_tok} tokens … ", end="", flush=True)

        # Clear buffers
        for l in all_moe_layers:
            selections[l] = []

        # Single prefill forward pass
        t0 = time.perf_counter()
        arr = mx.array(tokens, dtype=mx.uint32)
        cache = make_prompt_cache(model)
        out = model(arr[None], cache=cache)
        mx.eval(out)
        elapsed = time.perf_counter() - t0

        # Verify capture counts
        n_captured = len(selections[all_moe_layers[0]])
        print(f"{n_captured} tok captured  ({n_captured/elapsed:.0f} tok/s)", flush=True)

        # Build save dict: layer_L → (n_tok, K) int16
        train_n = int(n_captured * 0.8)
        test_n = n_captured - train_n
        save_dict: dict[str, np.ndarray] = {}
        for l in all_moe_layers:
            sel = selections[l]
            arr_l = np.stack(sel[:n_captured], axis=0).astype(np.int16)  # (n_tok, K)
            save_dict[f"layer_{l}"] = arr_l
        save_dict["meta_n_tokens"] = np.array([n_captured], dtype=np.int32)
        save_dict["meta_train_n"] = np.array([train_n], dtype=np.int32)
        save_dict["meta_test_n"] = np.array([test_n], dtype=np.int32)

        np.savez_compressed(out_path, **save_dict)
        print(f"    → saved {out_path}  (train={train_n}, test={test_n})", flush=True)

        topic_meta.append({"id": topic_idx, "name": name, f"{args.tag}_tokens": n_captured,
                            "train_n": train_n, "test_n": test_n})

    # Update topics.json
    topics_json = Path(args.out_dir) / "topics.json"
    existing_meta: list[dict] = []
    if topics_json.exists():
        with open(topics_json) as f:
            existing_meta = json.load(f)
    existing_by_id = {t["id"]: t for t in existing_meta}
    for m in topic_meta:
        tid = m["id"]
        if tid in existing_by_id:
            existing_by_id[tid].update(m)
        else:
            existing_by_id[tid] = m
    with open(topics_json, "w") as f:
        json.dump(sorted(existing_by_id.values(), key=lambda x: x["id"]), f, indent=2)

    print(f"\nDone. topics.json updated → {topics_json}", flush=True)


if __name__ == "__main__":
    main()
