import genome
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np

class MotorType(Enum):
    PULSE = 1
    SINE = 2

class Motor:
    def __init__(self, control_waveform, control_amp, control_freq, evolvable=True):
        self.evolvable = evolvable  # New flag to make motor properties evolvable
        if control_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        self.amp = control_amp
        self.freq = control_freq
        self.phase = 0

    def evolve_parameters(self, mutation_rate=0.1):
        if self.evolvable:
            self.amp += np.random.uniform(-mutation_rate, mutation_rate)
            self.freq += np.random.uniform(-mutation_rate, mutation_rate)
            self.amp = max(0, self.amp)  # Ensure amplitude remains non-negative
            self.freq = max(0.01, self.freq)  # Avoid zero frequency


    def get_output(self):
        self.phase = (self.phase + self.freq) % (np.pi * 2)
        if self.motor_type == MotorType.PULSE:
            if self.phase < np.pi:
                output = 1
            else:
                output = -1

        if self.motor_type == MotorType.SINE:
            output = np.sin(self.phase)

        #print (f"Information of Motor: {output}")
        return output

class Creature:
    def __init__(self, gene_count, fixed_parts=None):
        self.spec = genome.Genome.get_gene_spec()
        #print ("Gene Spec: ", self.spec)
        self.dna = genome.Genome.get_random_genome(len(self.spec), gene_count)
        #print ("DNA: ", self.dna)
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        self.fixed_parts = fixed_parts if fixed_parts else []  # List of part names to fix

    def get_expanded_links(self):
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links

        exp_links = [self.flat_links[0]]
        for link in self.flat_links[1:]:
            if link.name in self.fixed_parts:
                link.evolvable = False  # Make link fixed
            exp_links.append(link)
        self.exp_links = exp_links
        return self.exp_links


    def get_flat_links(self):
        if self.flat_links is None:
            gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)  # Updated to use genome_to_links
        return self.flat_links


    def to_xml(self):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        for link in self.exp_links:
            robot_tag.appendChild(link.to_link_element(adom))
        first = True
        for link in self.exp_links:
            if first:# skip the root node!
                first = False
                continue
            robot_tag.appendChild(link.to_joint_element(adom))
        robot_tag.setAttribute("name", "climber") #  choose a name!
        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        self.get_expanded_links()
        if self.motors == None:
            motors = []
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                m = Motor(l.control_waveform, l.control_amp,  l.control_freq)
                motors.append(m)
            self.motors = motors
        return self.motors

    def update_position(self, pos):
        if self.start_position == None:
            self.start_position = pos
        else:
            self.last_position = pos

    def get_distance_travelled(self):
        if self.start_position is None or self.last_position is None:
            return 0
        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        dist = np.linalg.norm(p1-p2)
        return dist

    def update_dna(self, dna):
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
