#!/usr/bin/env python3
from seedemu.compiler import Docker
from seedemu.core import Emulator
from seedemu.layers import ScionBase, ScionRouting, ScionIsd, Scion, Ospf
from seedemu.layers.Scion import LinkType as ScLinkType
from textwrap import dedent

# Global counter for cross-connect links.
link_counter = 1

# Bandwidth values for cross-connect links.
bandwidth = 2000000  # 2 Mbps

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
csA = asA.createControlService('cs_1'); csA.joinNetwork('net0')

# AS B (112)
asB = base.createAutonomousSystem(112)
scion_isd.addIsdAs(1, 112, is_core=True)
asB.createNetwork('net0').setDefaultLinkProperties(latency=0, bandwidth=0, packetDrop=0).setMtu(1400)
csB = asB.createControlService('cs_1'); csB.joinNetwork('net0')

# AS C (113)
asC = base.createAutonomousSystem(113)
scion_isd.addIsdAs(1, 113, is_core=True)
asC.createNetwork('net0').setDefaultLinkProperties(latency=0, bandwidth=0, packetDrop=0).setMtu(1400)
csC = asC.createControlService('cs_1'); csC.joinNetwork('net0')

# AS D (114)
asD = base.createAutonomousSystem(114)
scion_isd.addIsdAs(1, 114, is_core=True)
asD.createNetwork('net0').setDefaultLinkProperties(latency=0, bandwidth=0, packetDrop=0).setMtu(1400)
csD = asD.createControlService('cs_1'); csD.joinNetwork('net0')


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
# Raise SCION beacon policy caps to show more core segments (BestSetSize=40) ---
# ----------------------------
css = {111: csA, 112: csB, 113: csC, 114: csD}
# --- Raise beaconing caps so more core segments are kept/registered ---
propagation_yml = dedent("""
Type: "Propagation"
BestSetSize: 40
CandidateSetSize: 1000
""").strip()

core_reg_yml = dedent("""
Type: "CoreSegmentRegistration"
BestSetSize: 40
CandidateSetSize: 1000
""").strip()

enable_policies_sh = dedent(r'''#!/bin/sh
set -eu
mkdir -p /etc/scion/policies

TOML=/etc/scion/cs_1.toml
PROP=/etc/scion/policies/propagation.yml
CORE=/etc/scion/policies/core-reg.yml

# Ensure [beaconing] exists
if ! grep -q '^\[beaconing\]$' "$TOML"; then
  printf "\n[beaconing]\n" >> "$TOML"
fi

# Ensure [beaconing.policies] exists; if it exists, we will rewrite the block cleanly.
if grep -q '^\[beaconing\.policies\]$' "$TOML"; then
  # Rewrite the whole [beaconing.policies] block in-place
  awk -v PROP="$PROP" -v CORE="$CORE" '
    BEGIN{inpol=0}
    /^\[beaconing\.policies\]$/ { 
      print "[beaconing.policies]"
      print "propagation = \"" PROP "\""
      print "core_registration = \"" CORE "\""
      print "up_registration = \"\""
      print "down_registration = \"\""
      inpol=1
      next
    }
    /^\[/ { inpol=0 }     # any new section ends deletion
    { if (!inpol) print }
  ' "$TOML" > "$TOML.tmp" && mv "$TOML.tmp" "$TOML"
else
  # Insert the block immediately after [beaconing]
  awk -v PROP="$PROP" -v CORE="$CORE" '
    { print }
    /^\[beaconing\]$/ && !done {
      print "[beaconing.policies]"
      print "propagation = \"" PROP "\""
      print "core_registration = \"" CORE "\""
      print "up_registration = \"\""
      print "down_registration = \"\""
      done=1
    }
  ' "$TOML" > "$TOML.tmp" && mv "$TOML.tmp" "$TOML"
fi

# Show what we set (handy for debugging)
echo "---- beaconing.policies in $TOML ----"
awk '/^\[beaconing.policies\]$/,/^\[/' "$TOML" | sed '/^\[/! s/^/  /'
''').strip()

def wire_beacon_policies(cs):
    # Drop policy files
    cs.setFile('/etc/scion/policies/propagation.yml', propagation_yml)
    cs.setFile('/etc/scion/policies/core-reg.yml',   core_reg_yml)
    # Drop and run a tiny boot script that patches control.toml
    cs.setFile('/usr/local/bin/enable-policies.sh', enable_policies_sh)
    cs.appendStartCommand('chmod +x /usr/local/bin/enable-policies.sh && /usr/local/bin/enable-policies.sh')

for asn, cs in css.items():
    wire_beacon_policies(cs)


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
