Current test:

Reducing image size, 14 API calls total now at constant cost

This seems to come to 284 dollars which is more palatable for the whole set, we will need to expand the input prompt though
May be able to compress image more.

PROMPT TEMPLATE:

Overview
This document provides guidelines for using a comprehensive tagging schema designed for classifying artwork, particularly character art for tabletop roleplaying games, though it can also be applied to a wider range of artworks including memes and other digital creations. This schema is intended to be used within a system prompt for a large language model to facilitate the organization and retrieval of artworks.

Schema Structure
The tagging system is hierarchical, allowing for general categorization at the top level, with increasing specificity at each subsequent level. Tags are appended to base tags with forward slashes (e.g., Creature/Humanoid/Elf).

Primary Categories
Creature: Classification based on the type of creature depicted in the artwork.
Size: The relative size of creatures or objects within the artwork.
Sex: Physical sex as represented visually in the artwork.
Genre: The artistic or thematic style of the artwork.
Things: Objects or non-sentient subjects of the artwork.
Composition: The arrangement or type of scene depicted.
Mood/Emotion: The emotional tone or mood conveyed by the artwork.
Color Palette: Dominant color themes in the artwork.
Setting/Environment: The location or setting where the artwork is placed.
Technique/Style: The method or style used in the creation of the artwork.
Other: Additional aspects not covered by the above categories.
Instructions for Use
Tags should be used to describe the content of the artwork as accurately and specifically as possible.
All applicable tags should be included. For instance, if an artwork features a humanoid that appears reptilian, it should be tagged with both Creature/Humanoid and Creature/Reptilian.
If specificity within a category cannot be determined, use only the highest certain level of specificity. For example, if a humanoid species is unclear, use Creature/Humanoid.
If an artwork features elements that are not covered by the existing tags but are widely recognizable, such as specific species from popular media (e.g., Asari from Mass Effect), include these as subtags.
The tags are not exhaustive. If a new category or subcategory is identified, it should be added to the schema following the existing hierarchical format.
In cases of uncertainty or when the artwork is ambiguous, it is acceptable to omit certain levels of specificity or to use the 'Other' category for tagging.
Example Tagging
An artwork depicting a lone human warrior in a dark, futuristic setting, holding a weapon, might be tagged as:

Creature/Humanoid/Human
Sex/Male
Genre/Sci-fi
Things/Weapon
Composition/Solo
Mood/Emotion/Dark
Setting/Environment/Futuristic
Technique/Style/Digital
Conclusion
This schema is a living system, designed to evolve as new artwork is created and added to the collection. Users are encouraged to maintain the integrity of the schema while also being open to its evolution, ensuring that it remains a dynamic and useful tool for the classification and retrieval of artwork.

-Creature
--Humanoid
---Human
---Elf
---Bestial
---Goblin
---Merfolk
---Orc
--Aquatic
--Arthropod
--Avian
--Reptilian
--Amphibian
--Amorphous
--Robotic
--Undead
--Elemental
--Corrupted
-Size
--Small
--Medium
--Large
-Sex
--Male
--Female
--Androgynous
-Genre
--Fantasy
--Modern
--Sci-fi
--Horror
--Steampunk
--Historical
-Things
--Vehicles
--Equipment
--Food
--Items
--Landscapes
-Composition
--Group
--Solo
--Action
--Portrait
--Landscape
-Mood
--Serious
--Humorous
--Dark
--Light-hearted
-Color
--Monochrome
--Vibrant
--Dark_Tones
--Pastel
--Earth_Tones
-Environment
--Urban
--Rural
--Dungeon
--Space
--Forest
--Mountain
-Style
--Sketch
--Digital
--Oil_Painting
--Watercolor
--Charcoal
-Other
--HasText
--ConceptArt
--Meme
