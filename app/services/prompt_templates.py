from __future__ import annotations

from app.schemas import PromptTemplate


PROMPT_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        id="kitchen-refurb",
        title="Kitchen refurbishment",
        description="Strip out, first fix, finishes, joinery, flooring, and final decorations.",
        category="Interiors",
        prompt=(
            "PROPERTY:\n"
            "Type: Flat\n"
            "\n"
            "FULL SCOPE:\n"
            "Kitchen refurbishment\n"
            "\n"
            "FLOOR AREAS & ROOMS:\n"
            "Ground Floor\n"
            "Kitchen: 4.5 x 3.2\n"
            "\n"
            "DEMOLITION:\n"
            "Strip out old cabinets and worktops\n"
            "Remove wall tiles and floor finish\n"
            "\n"
            "PLASTERING:\n"
            "Skim walls where required\n"
            "\n"
            "ELECTRICAL:\n"
            "Kitchen: 8 downlights\n"
            "Kitchen: 6 double sockets\n"
            "Kitchen: 3 appliance hardwire connections\n"
            "\n"
            "PLUMBING:\n"
            "Reconnect sink\n"
            "Reconnect appliances\n"
            "\n"
            "FLOORING:\n"
            "Kitchen: Engineered timber\n"
            "\n"
            "DECORATING:\n"
            "Paint kitchen walls and ceiling\n"
        ),
    ),
    PromptTemplate(
        id="bathroom-renovation",
        title="Bathroom renovation",
        description="Demolition, plumbing and electrics, waterproofing, tiling, sanitaryware, final finishes.",
        category="Interiors",
        prompt=(
            "PROPERTY:\n"
            "Type: Flat\n"
            "\n"
            "FULL SCOPE:\n"
            "Bathroom renovation\n"
            "\n"
            "FLOOR AREAS & ROOMS:\n"
            "Ground Floor\n"
            "Bathroom: 2.5 x 2.0\n"
            "\n"
            "DEMOLITION:\n"
            "Strip out existing sanitaryware\n"
            "Hack off wall and floor tiles\n"
            "\n"
            "PLUMBING:\n"
            "Install WC\n"
            "Install basin\n"
            "Install shower enclosure\n"
            "\n"
            "ELECTRICAL:\n"
            "Bathroom: 2 downlights\n"
            "Bathroom: extractor fan with ducting\n"
            "\n"
            "TILING:\n"
            "Full-height wall tiling to bathroom\n"
            "Floor tiling to bathroom\n"
            "\n"
            "DECORATING:\n"
            "Paint bathroom ceiling\n"
        ),
    ),
    PromptTemplate(
        id="full-flat-refresh",
        title="Full flat refresh",
        description="Decorations, flooring, joinery touch-ups, small electrical and plumbing updates.",
        category="Whole property",
        prompt=(
            "PROPERTY:\n"
            "Type: Flat\n"
            "\n"
            "FULL SCOPE:\n"
            "Full flat refresh\n"
            "\n"
            "FLOOR AREAS & ROOMS:\n"
            "Ground Floor\n"
            "Living Room: 5.5 x 4.0\n"
            "Kitchen: 4.0 x 2.8\n"
            "Bedroom 1: 4.0 x 3.5\n"
            "Bedroom 2: 3.5 x 3.0\n"
            "Bathroom: 2.5 x 2.0\n"
            "Hallway: 4.0 x 1.2\n"
            "\n"
            "PLASTERING:\n"
            "Patch and make good walls throughout\n"
            "\n"
            "ELECTRICAL:\n"
            "Replace light fittings throughout\n"
            "Add sockets where needed\n"
            "\n"
            "FLOORING:\n"
            "Living Room: Engineered timber\n"
            "Kitchen: Engineered timber\n"
            "Bedroom 1: Carpet\n"
            "Bedroom 2: Carpet\n"
            "Hallway: Engineered timber\n"
            "\n"
            "DECORATING:\n"
            "Paint walls and ceilings throughout\n"
            "\n"
            "WOODWORK PAINTING:\n"
            "Paint skirting and architraves throughout\n"
        ),
    ),
    PromptTemplate(
        id="loft-conversion-scope",
        title="Loft conversion scope",
        description="Structural works, insulation, partitions, stairs, MEP, plastering and final finishes.",
        category="Extensions",
        prompt=(
            "PROPERTY:\n"
            "Type: Terraced house\n"
            "\n"
            "FULL SCOPE:\n"
            "Loft conversion\n"
            "\n"
            "FLOOR AREAS & ROOMS:\n"
            "Loft\n"
            "Loft Bedroom: 6.0 x 4.5\n"
            "Loft Ensuite: 2.5 x 1.8\n"
            "\n"
            "STRUCTURE:\n"
            "Rear dormer: 4.5 x 3.5\n"
            "Structural opening and steel beam\n"
            "New floor structure\n"
            "\n"
            "WINDOWS:\n"
            "2 x Velux (MK04)\n"
            "\n"
            "CEILINGS & INSULATION:\n"
            "120 mm between rafters\n"
            "50 mm insulated board under\n"
            "\n"
            "CARPENTRY:\n"
            "Stud partitions to loft\n"
            "\n"
            "ELECTRICAL:\n"
            "Loft Bedroom: 6 downlights\n"
            "Loft Bedroom: 4 double sockets\n"
            "Loft Ensuite: 2 downlights\n"
            "Loft Ensuite: extractor fan with ducting\n"
            "\n"
            "PLUMBING:\n"
            "Install basin in ensuite\n"
            "Install WC in ensuite\n"
            "Install shower in ensuite\n"
            "\n"
            "PLASTERING:\n"
            "Skim walls and ceilings throughout loft\n"
            "\n"
            "FLOORING:\n"
            "Loft Bedroom: Carpet\n"
            "Loft Ensuite: Floor tiles\n"
            "\n"
            "DECORATING:\n"
            "Paint loft walls and ceilings\n"
        ),
    ),
]


def get_prompt_templates() -> list[PromptTemplate]:
    return PROMPT_TEMPLATES
