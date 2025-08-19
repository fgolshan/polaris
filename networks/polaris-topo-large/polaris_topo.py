#!/usr/bin/env python3
from seedemu.compiler import Docker
from seedemu.core import Emulator
from seedemu.layers import ScionBase, ScionRouting, ScionIsd, Scion, Ospf
from seedemu.layers.Scion import LinkType as ScLinkType

# Global counter for cross-connect links.
link_counter = 1

# Bandwidth values for cross-connect links.
bandwidth = 5000000  # 5 Mbps

def next_ips():
    """
    Allocates a unique /29 block for a cross-connect and returns a pair of IP addresses.
    We always use the 2nd and 3rd addresses of the block (e.g., x.x.x.2/29 and x.x.x.3/29).
    """
    global link_counter
    # Each /29 block contains 8 addresses.
    offset = (link_counter - 1) * 8
    third_octet = offset // 256
    fourth_octet = offset % 256
    local_ip = f"10.1.{third_octet}.{fourth_octet + 2}/29"
    remote_ip = f"10.1.{third_octet}.{fourth_octet + 3}/29"
    link_counter += 1
    return local_ip, remote_ip

# Initialize the emulator and layers.
emu = Emulator()
base = ScionBase()
routing = ScionRouting()
scion_isd = ScionIsd()
scion = Scion()
ospf = Ospf()

# Create ISD 1.
base.createIsolationDomain(1)

# ----------------------------
# Create Autonomous Systems (all core)
# ----------------------------

# AS A (111)
asA = base.createAutonomousSystem(111)
scion_isd.addIsdAs(1, 111, is_core=True)
asA.createNetwork('net0').setDefaultLinkProperties(latency=0, bandwidth=0, packetDrop=0).setMtu(1400)
asA.createControlService('cs_1').joinNetwork('net0')

# AS B (112)
asB = base.createAutonomousSystem(112)
scion_isd.addIsdAs(1, 112, is_core=True)
asB.createNetwork('net0').setDefaultLinkProperties(latency=0, bandwidth=0, packetDrop=0).setMtu(1400)
asB.createControlService('cs_1').joinNetwork('net0')

# AS C (113)
asC = base.createAutonomousSystem(113)
scion_isd.addIsdAs(1, 113, is_core=True)
asC.createNetwork('net0').setDefaultLinkProperties(latency=0, bandwidth=0, packetDrop=0).setMtu(1400)
asC.createControlService('cs_1').joinNetwork('net0')

# AS D (114)
asD = base.createAutonomousSystem(114)
scion_isd.addIsdAs(1, 114, is_core=True)
asD.createNetwork('net0').setDefaultLinkProperties(latency=0, bandwidth=0, packetDrop=0).setMtu(1400)
asD.createControlService('cs_1').joinNetwork('net0')

# Mapping from AS number to its corresponding AS object.
ases = {
    111: asA,
    112: asB,
    113: asC,
    114: asD,
}

# ----------------------------
# Define Inter-AS Links
# ----------------------------
# Topology definition:
#   - A (111) <-> B (112): 3 links
#   - B (112) <-> C (113): 2 links
#   - C (113) <-> D (114): 3 links
#   - D (114) <-> A (111): 2 links (represented as (111,114))
#   - B (112) <-> D (114): 2 links
links = {
    (111, 112): 3,
    (112, 113): 2,
    (113, 114): 3,
    (111, 114): 2,
    (112, 114): 2,
}

# ----------------------------
# Create Cross-Connect Links using the Core link type
# ----------------------------
for (asn1, asn2), count in links.items():
    for i in range(1, count + 1):
        router_name = f"br_{asn1}_{asn2}_{i}"
        local_ip, remote_ip = next_ips()
        
        # Create border router in AS with lower number.
        router1 = ases[asn1].createRouter(router_name).joinNetwork('net0')
        router1.crossConnect(asn2, router_name, local_ip, latency=0, bandwidth=bandwidth, packetDrop=0, MTU=1280)
        
        # Create border router in AS with higher number.
        router2 = ases[asn2].createRouter(router_name).joinNetwork('net0')
        router2.crossConnect(asn1, router_name, remote_ip, latency=0, bandwidth=bandwidth, packetDrop=0, MTU=1280)
        
        # Add the SCION cross-connect link using the Core link type.
        scion.addXcLink((1, asn1), (1, asn2), ScLinkType.Core, a_router=router_name, b_router=router_name)

# ----------------------------
# Add layers and dump topology
# ----------------------------
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(scion_isd)
emu.addLayer(scion)
emu.addLayer(ospf)

# Dump the emulator topology to a file.
# emu.dump("./seed_mini.bin")

# Optional: render or compile the topology.
emu.render()
emu.compile(Docker(internetMapEnabled=True), './polaris_topo_out')
